"""Post-run accounting for raw action choices, paired changes, latency, and cost."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import returned_model_is_compatible
from .common import append_jsonl, read_jsonl, utc_now, write_json


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def _rows_by(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        final_rows = len(group)
        completed = [row for row in group if row.get("error") is None]
        selected = [row for row in completed if row.get("selected_action") is not None]
        correct = [row for row in selected if row.get("canonical_action_correct") is True]
        latency = [_number(row.get("latency_ms")) for row in completed]
        costs = [_number(row.get("usage", {}).get("cost")) for row in completed]
        known_costs = [value for value in costs if value is not None]
        summaries.append(
            {
                "group": list(group_key),
                "final_logical_request_rows": final_rows,
                "completed_rows": len(completed),
                "selected_action_rows": len(selected),
                "canonical_action_correct": len(correct),
                "selected_action_rate": round(len(selected) / final_rows, 6) if final_rows else None,
                "unconditional_canonical_accuracy": round(len(correct) / final_rows, 6) if final_rows else None,
                "canonical_accuracy_among_selected_actions": round(len(correct) / len(selected), 6) if selected else None,
                "valid_distribution_rows": sum(bool(row.get("distribution_valid")) for row in completed),
                "mean_latency_ms": _mean(value for value in latency if value is not None),
                "final_response_cost_usd_lower_bound": round(sum(known_costs), 10),
                "final_rows_with_unknown_cost": sum(value is None for value in costs),
                "final_response_cost_complete": not any(value is None for value in costs),
            }
        )
    return summaries


def _decision_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") == "C0":
            key = (str(row.get("model_arm")), str(row.get("task_id")), int(row.get("repeat", 0)))
            baseline[key] = row
    changes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("condition") == "C0":
            continue
        key = (str(row.get("model_arm")), str(row.get("task_id")), int(row.get("repeat", 0)))
        reference = baseline.get(key)
        paired = reference is not None and row.get("error") is None and reference.get("error") is None
        selected_changed = None
        p2_changed = None
        p3_changed = None
        if paired:
            if row.get("selected_action") is not None and reference.get("selected_action") is not None:
                selected_changed = row["selected_action"] != reference["selected_action"]
            p2_row = row.get("policies", {}).get("P2", {})
            p2_ref = reference.get("policies", {}).get("P2", {})
            if p2_row.get("action_id") is not None and p2_ref.get("action_id") is not None:
                p2_changed = p2_row["action_id"] != p2_ref["action_id"]
            p3_row = row.get("policies", {}).get("P3", {})
            p3_ref = reference.get("policies", {}).get("P3", {})
            if p3_row.get("action_id") is not None and p3_ref.get("action_id") is not None:
                p3_changed = p3_row["action_id"] != p3_ref["action_id"]
        changes.append(
            {
                "attempt_id": row.get("attempt_id"),
                "model_arm": row.get("model_arm"),
                "task_id": row.get("task_id"),
                "domain": row.get("domain"),
                "condition": row.get("condition"),
                "repeat": row.get("repeat"),
                "paired_with_c0": paired,
                "c0_attempt_id": reference.get("attempt_id") if reference else None,
                "selected_canonical_action_changed_from_c0": selected_changed,
                "P2_action_changed_from_c0": p2_changed,
                "P3_action_changed_from_c0": p3_changed,
                "P4_source_logical_request_id": row.get("policies", {}).get("P4", {}).get("source_logical_request_id"),
                "P4_link_matches_paired_c0": (
                    row.get("policies", {}).get("P4", {}).get("source_logical_request_id")
                    == reference.get("logical_request_id")
                    if reference is not None else None
                ),
                "selected_action": row.get("selected_action"),
                "c0_selected_action": reference.get("selected_action") if reference else None,
            }
        )
    return changes


def _repeat_disagreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("model_arm")), str(row.get("task_id")), str(row.get("condition")))].append(row)
    records: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        by_repeat = {row.get("repeat"): row for row in group}
        if set(by_repeat) != {1, 2}:
            continue
        first, second = by_repeat[1], by_repeat[2]
        first_choice, second_choice = first.get("selected_action"), second.get("selected_action")
        records.append(
            {
                "model_arm": key[0], "task_id": key[1], "condition": key[2],
                "comparable": first_choice is not None and second_choice is not None,
                "selected_canonical_action_disagrees": (
                    first_choice != second_choice if first_choice is not None and second_choice is not None else None
                ),
                "repeat_1_action": first_choice, "repeat_2_action": second_choice,
            }
        )
    return records


def _clarification_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("model_arm"))].append(row)
    results: list[dict[str, Any]] = []
    for model, group in sorted(groups.items()):
        ambiguous = [row for row in group if str(row.get("gold_action_id", "")).endswith("request_clarification")]
        actionable = [row for row in group if row not in ambiguous]
        ambiguous_selected = [row for row in ambiguous if row.get("selected_action") is not None and row.get("error") is None]
        actionable_selected = [row for row in actionable if row.get("selected_action") is not None and row.get("error") is None]
        clarification_on_ambiguous = sum(
            str(row.get("selected_action", "")).endswith("request_clarification") for row in ambiguous
        )
        nonclarification_on_actionable = sum(
            not str(row.get("selected_action", "")).endswith("request_clarification")
            for row in actionable
            if row.get("selected_action") is not None and row.get("error") is None
        )
        results.append(
            {
                "model_arm": model,
                "ambiguous_task_rows": len(ambiguous),
                "ambiguous_task_selected_action_rows": len(ambiguous_selected),
                "clarification_selected_on_ambiguous_task": clarification_on_ambiguous,
                "raw_clarification_selection_rate_on_ambiguous_tasks": round(
                    clarification_on_ambiguous / len(ambiguous), 6
                ) if ambiguous else None,
                "clarification_rate_among_selected_ambiguous_tasks": round(
                    clarification_on_ambiguous / len(ambiguous_selected), 6
                ) if ambiguous_selected else None,
                "action_required_rows": len(actionable),
                "action_required_selected_action_rows": len(actionable_selected),
                "nonclarification_selected_on_action_required_tasks": nonclarification_on_actionable,
                "raw_nonclarification_selection_rate_on_action_required_tasks": round(
                    nonclarification_on_actionable / len(actionable), 6
                ) if actionable else None,
                "nonclarification_rate_among_selected_action_required_tasks": round(
                    nonclarification_on_actionable / len(actionable_selected), 6
                ) if actionable_selected else None,
            }
        )
    return results


def analyze_run(run_dir: Path) -> dict[str, Any]:
    ledger_path = run_dir / "ledger.jsonl"
    physical_rows = read_jsonl(ledger_path)
    # Retain every transport attempt in ledger.jsonl, but analyze only the final
    # result for each planned logical request so retries do not inflate N.
    rows = [row for row in physical_rows if row.get("is_final_attempt", True)]
    decision_rows = _decision_changes(rows)
    decisions_path = run_dir / "decision_changes.jsonl"
    if decisions_path.exists():
        decisions_path.unlink()
    for record in decision_rows:
        append_jsonl(decisions_path, record)
    repeat_rows = _repeat_disagreement(rows)
    completed = [row for row in rows if row.get("error") is None]
    physical_error_attempts = [row for row in physical_rows if row.get("error") is not None]
    retry_attempts = [row for row in physical_rows if int(row.get("transport_attempt", 0)) > 0]
    finalized_logical_ids = {
        str(row["logical_request_id"])
        for row in rows
        if isinstance(row.get("logical_request_id"), str)
    }
    physical_costs = [_number(row.get("usage", {}).get("cost")) for row in physical_rows]
    known_physical_costs = [value for value in physical_costs if value is not None]
    snapshots: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        snapshots[
            (
                row.get("model_arm"),
                row.get("requested_model"),
                row.get("returned_model"),
                row.get("provider"),
                row.get("system_fingerprint"),
            )
        ] += 1
    snapshot_records = [
        {
            "model_arm": key[0],
            "requested_model": key[1],
            "returned_model": key[2],
            "provider": key[3],
            "system_fingerprint": key[4],
            "final_rows": count,
        }
        for key, count in sorted(snapshots.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]
    model_returned_ids: dict[str, set[str]] = defaultdict(set)
    model_providers: dict[str, set[str]] = defaultdict(set)
    for row in completed:
        if isinstance(row.get("returned_model"), str):
            model_returned_ids[str(row["model_arm"])].add(row["returned_model"])
        if isinstance(row.get("provider"), str):
            model_providers[str(row["model_arm"])].add(row["provider"])
    comparability_flags: list[str] = []
    for record in completed:
        if record.get("returned_model") is not None and not returned_model_is_compatible(record.get("requested_model", ""), record["returned_model"]):
            comparability_flags.append(
                f"returned-model mismatch for {record['model_arm']}: {record['returned_model']} != {record['requested_model']}"
            )
    for model_arm in sorted(set(model_returned_ids) | set(model_providers)):
        if len(model_returned_ids[model_arm]) > 1 or len(model_providers[model_arm]) > 1:
            comparability_flags.append(
                f"multiple provider/model snapshots observed for {model_arm}; stratify or exclude mixed snapshots before comparison"
            )
    for field in ("returned_model",):
        affected_arms = sorted(
            {
                str(row.get("model_arm"))
                for row in rows
                if row.get("error") is None and row.get(field) is None
            }
        )
        if affected_arms:
            comparability_flags.append(
                f"successful response(s) for {', '.join(affected_arms)} lack {field}; snapshot provenance is unverifiable"
            )
    provenance_notes = []
    if any(row.get("error") is None and row.get("provider") is None for row in rows):
        provenance_notes.append("Some successful responses lack optional returned provider metadata; the request's provider constraint remains in the raw payload.")
    if any(row.get("error") is None and row.get("system_fingerprint") is None for row in rows):
        provenance_notes.append("Some successful responses lack optional system_fingerprint; returned model remains logged.")
    manifest_path = run_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned_requests = manifest.get("planned_requests") if isinstance(manifest, dict) else None
    incomplete_logical_requests = (
        max(0, planned_requests - len(finalized_logical_ids))
        if isinstance(planned_requests, int) and not isinstance(planned_requests, bool)
        else None
    )
    summary = {
        "analysis_schema_version": 1,
        "generated_at": utc_now(),
        "run_directory": str(run_dir),
        "manifest": manifest,
        "request_accounting": {
            "planned_requests": planned_requests,
            "attempt_rows_written": len(physical_rows),
            "final_logical_request_rows": len(rows),
            "completed_without_transport_or_parse_error": len(completed),
            "final_logical_request_errors": len(rows) - len(completed),
            "physical_error_attempts": len(physical_error_attempts),
            "retry_attempts": len(retry_attempts),
            "incomplete_logical_requests": incomplete_logical_requests,
            "valid_distributions": sum(bool(row.get("distribution_valid")) for row in completed),
            "known_actual_cost_usd_lower_bound": round(sum(known_physical_costs), 10),
            "physical_rows_with_unknown_cost": sum(value is None for value in physical_costs),
            "actual_cost_complete": not any(value is None for value in physical_costs),
        },
        "by_model_condition_domain": _rows_by(
            rows, lambda row: (row.get("model_arm"), row.get("condition"), row.get("domain"))
        ),
        "by_model_condition": _rows_by(rows, lambda row: (row.get("model_arm"), row.get("condition"))),
        "decision_changes": {
            "paired_non_c0_rows": sum(bool(row["paired_with_c0"]) for row in decision_rows),
            "selected_canonical_action_changes": sum(
                row["selected_canonical_action_changed_from_c0"] is True for row in decision_rows
            ),
            "P2_action_changes": sum(row["P2_action_changed_from_c0"] is True for row in decision_rows),
            "P3_action_changes": sum(row["P3_action_changed_from_c0"] is True for row in decision_rows),
            "P4_links_matching_paired_c0": sum(row["P4_link_matches_paired_c0"] is True for row in decision_rows),
            "records_path": str(decisions_path),
        },
        "repeat_disagreement": {
            "comparable_pairs": sum(bool(row["comparable"]) for row in repeat_rows),
            "selected_canonical_action_disagreements": sum(
                row["selected_canonical_action_disagrees"] is True for row in repeat_rows
            ),
            "records": repeat_rows,
        },
        "raw_clarification_behavior": _clarification_behavior(rows),
        "response_snapshots": snapshot_records,
        "comparability_flags": comparability_flags,
        "provenance_notes": provenance_notes,
        "policy_note": (
            "P1/P2/P3 are derived from each raw response. No coverage, selective-risk, or accept/defer claim is computed "
            "here because thresholds are fitted only on the later C0 calibration split. P4 is a C0 canonical-request reuse, not a new call."
        ),
        "synthetic_run": bool(rows) and all(bool(row.get("synthetic")) for row in rows),
        "contains_synthetic_rows": any(bool(row.get("synthetic")) for row in rows),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing ledger.jsonl")
    args = parser.parse_args(argv)
    summary = analyze_run(args.run)
    print(f"Analyzed {summary['request_accounting']['attempt_rows_written']} request rows: {args.run / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
