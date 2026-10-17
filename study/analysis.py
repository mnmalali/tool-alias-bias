"""Prespecified calibration and held-out analysis of the 300-task matrix."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from pilot.common import read_json, read_jsonl, utc_now, write_json
from pilot.registry import CONDITIONS
from .figures import write_figures


BOOTSTRAP_SEED = 20261009
TARGET_COVERAGE = 0.8
PRACTICAL_COVERAGE_SHIFT = 0.10
MIN_PARTIAL_CALIBRATION_TASKS = 80


def _fit_threshold(scores: list[float]) -> dict[str, Any]:
    """Choose the attainable >= cutoff nearest 80%, preferring lower coverage on a tie."""
    if not scores or any(not math.isfinite(score) for score in scores):
        raise ValueError("Calibration scores are missing or nonfinite")
    candidates = [math.nextafter(max(scores), math.inf), *sorted(set(scores), reverse=True)]
    choices = []
    for cutoff in candidates:
        accepted = sum(score >= cutoff for score in scores)
        # Target is exactly 4/5. Integer distance preserves an exact tie.
        choices.append((abs(5 * accepted - 4 * len(scores)), accepted, -cutoff, cutoff))
    _, accepted, _, cutoff = min(choices)
    coverage = accepted / len(scores)
    return {
        "cutoff": cutoff,
        "accept_rule": "score >= cutoff",
        "target_coverage": TARGET_COVERAGE,
        "attained_coverage": coverage,
        "calibration_requests": len(scores),
        "tie_rule": "If several attainable coverages are equally close to 0.8, choose the lower coverage.",
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def _cluster_samples(tasks: list[dict[str, Any]], replicates: int) -> list[list[str]]:
    """Sample scenario families within domain and retain all prompts in each family."""
    families: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        families[task["domain"]][task["scenario_family_id"]].append(task["task_id"])
    generator = random.Random(BOOTSTRAP_SEED)
    samples: list[list[str]] = []
    for _ in range(replicates):
        sample: list[str] = []
        for domain in sorted(families):
            family_ids = sorted(families[domain])
            for _ in family_ids:
                family_id = generator.choice(family_ids)
                sample.extend(families[domain][family_id])
        samples.append(sample)
    return samples


def _interval(values: dict[str, float], samples: list[list[str]]) -> dict[str, float]:
    estimate = mean(values.values())
    draws = sorted(mean(values[task_id] for task_id in sample) for sample in samples)
    return {
        "estimate": estimate,
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accepted = [row for row in rows if row["accepted"]]
    correct = sum(row["correct"] for row in rows)
    accepted_correct = sum(row["correct"] for row in accepted)
    accepted_wrong = len(accepted) - accepted_correct
    briers = [row["brier"] for row in rows if row["brier"] is not None]
    return {
        "requests": n,
        "independent_tasks": len({row["task_id"] for row in rows}),
        "scenario_families": len({row["scenario_family_id"] for row in rows}),
        "unconditional_canonical_action_accuracy": correct / n if n else None,
        "coverage": len(accepted) / n if n else None,
        "selective_risk": accepted_wrong / len(accepted) if accepted else None,
        "accepted_wrong_rate": accepted_wrong / n if n else None,
        "accepted_correct_yield": accepted_correct / n if n else None,
        "accepted_requests": len(accepted),
        "accepted_wrong_requests": accepted_wrong,
        "accepted_correct_requests": accepted_correct,
        "mean_gating_score": mean(row["score"] for row in rows) if n else None,
        "mean_action_distribution_brier": mean(briers) if briers else None,
    }


def _brier(row: dict[str, Any]) -> float | None:
    masses = row.get("policies", {}).get("P3", {}).get("action_masses")
    if not isinstance(masses, dict):
        return None
    gold = row["gold_action_id"]
    return sum((float(probability) - (1 if action_id == gold else 0)) ** 2 for action_id, probability in masses.items())


def _policy_source(row: dict[str, Any], policy: str, baseline: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    if policy == "P4":
        return baseline[(row["model_arm"], row["task_id"], row["repeat"])]["policies"]["P2"]
    return row["policies"][policy]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _operational_accounting(plan: list[dict[str, Any]], physical: list[dict[str, Any]],
                            finals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"plan": [], "physical": [], "final": []})
    for source, rows in (("plan", plan), ("physical", physical), ("final", finals)):
        for row in rows:
            groups[(row["split"], row["model_arm"], row["condition"])][source].append(row)
    table: list[dict[str, Any]] = []
    for (split, model, condition), grouped in sorted(groups.items()):
        attempts = grouped["physical"]
        final = grouped["final"]
        latencies = [float(row["latency_ms"]) for row in attempts if isinstance(row.get("latency_ms"), (int, float))]
        input_tokens = [row.get("usage", {}).get("input_tokens", row.get("usage", {}).get("prompt_tokens")) for row in attempts]
        output_tokens = [row.get("usage", {}).get("output_tokens", row.get("usage", {}).get("completion_tokens")) for row in attempts]
        costs = [row.get("usage", {}).get("cost") for row in attempts]
        valid_input = [float(value) for value in input_tokens if isinstance(value, (int, float)) and not isinstance(value, bool)]
        valid_output = [float(value) for value in output_tokens if isinstance(value, (int, float)) and not isinstance(value, bool)]
        valid_costs = [float(value) for value in costs if isinstance(value, (int, float)) and not isinstance(value, bool)]
        table.append({
            "split": split, "model_arm": model, "condition": condition,
            "planned_logical_requests": len(grouped["plan"]),
            "final_logical_requests": len(final),
            "missing_logical_requests": len(grouped["plan"]) - len(final),
            "valid_final_distributions": sum(row.get("error") is None and bool(row.get("distribution_valid")) for row in final),
            "final_error_or_invalid_requests": sum(row.get("error") is not None or not row.get("distribution_valid") for row in final),
            "physical_attempts": len(attempts),
            "retry_attempts": sum(int(row.get("transport_attempt", 0)) > 0 for row in attempts),
            "input_tokens_known_total": sum(valid_input),
            "input_token_unknown_attempts": len(input_tokens) - len(valid_input),
            "output_tokens_known_total": sum(valid_output),
            "output_token_unknown_attempts": len(output_tokens) - len(valid_output),
            "cost_usd_known_total": sum(valid_costs),
            "cost_unknown_attempts": len(costs) - len(valid_costs),
            "physical_latency_mean_ms": mean(latencies) if latencies else None,
            "physical_latency_median_ms": median(latencies) if latencies else None,
        })
    return table


def _tie_rates(finals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in finals:
        if row.get("error") is None and row.get("distribution_valid"):
            groups[(row["split"], row["model_arm"], row["condition"])].append(row)
    diagnostics: list[dict[str, Any]] = []
    for (split, model, condition), rows in sorted(groups.items()):
        p2_ties = p2_cross_action = p2_low_high_action_change = p3_ties = 0
        observed = 0
        for row in rows:
            probabilities = row.get("option_probabilities")
            masses = row.get("policies", {}).get("P3", {}).get("action_masses")
            if not isinstance(probabilities, dict) or not isinstance(masses, dict) or not probabilities or not masses:
                continue
            mapping = {entry["option_id"]: entry["action_id"] for entry in row.get("catalog_mapping", [])}
            maximum = max(float(value) for value in probabilities.values())
            tied_options = sorted(option_id for option_id, value in probabilities.items() if float(value) == maximum)
            tied_actions = {mapping[option_id] for option_id in tied_options if option_id in mapping}
            p2_ties += len(tied_options) > 1
            p2_cross_action += len(tied_actions) > 1
            if len(tied_options) > 1 and tied_options[0] in mapping and tied_options[-1] in mapping:
                p2_low_high_action_change += mapping[tied_options[0]] != mapping[tied_options[-1]]
            highest_mass = max(float(value) for value in masses.values())
            p3_ties += sum(float(value) == highest_mass for value in masses.values()) > 1
            observed += 1
        diagnostics.append({
            "split": split, "model_arm": model, "condition": condition,
            "valid_distributions_examined": observed,
            "P2_exact_visible_argmax_tie_requests": p2_ties,
            "P2_exact_visible_argmax_tie_rate": p2_ties / observed if observed else None,
            "P2_ties_crossing_actions": p2_cross_action,
            "P2_lowest_vs_highest_tied_ID_changes_action": p2_low_high_action_change,
            "P3_exact_action_mass_argmax_tie_requests": p3_ties,
            "P3_exact_action_mass_argmax_tie_rate": p3_ties / observed if observed else None,
        })
    return diagnostics


def _task_mean(index: dict[tuple[str, str, str, str, int], dict[str, Any]], model: str, policy: str,
               task_id: str, condition: str, field: str) -> float:
    return mean(float(index[(model, policy, task_id, condition, repeat)][field]) for repeat in (1, 2))


def analyze_study(run_dir: Path, *, bootstrap_replicates: int = 2000) -> dict[str, Any]:
    if bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    manifest = read_json(run_dir / "manifest.json")
    plan = read_jsonl(run_dir / "plan.jsonl")
    physical = read_jsonl(run_dir / "ledger.jsonl")
    finals = [row for row in physical if row.get("is_final_attempt", True)]
    final_by_id = {row.get("logical_request_id"): row for row in finals}
    plan_ids = {row.get("logical_request_id") for row in plan}
    if len(plan) != 10800 or len(plan_ids) != 10800 or not manifest.get("full_300x6x2x3_matrix"):
        raise ValueError("Study analysis requires a complete planned 300 x 6 x 2 x 3 matrix")
    if len(final_by_id) != len(finals) or set(final_by_id) - plan_ids:
        raise ValueError("Final ledger has duplicate or unknown logical request IDs")
    missing = sorted(plan_ids - set(final_by_id))
    errors = [row for row in finals if row.get("error") is not None or not row.get("distribution_valid")]
    synthetic = [row for row in finals if row.get("synthetic")]
    mixed_synthetic = bool(synthetic) and len(synthetic) != len(finals)
    completed = not missing and not errors and not mixed_synthetic
    status = "incomplete" if not completed else ("synthetic" if synthetic else "complete")
    pending_review_count = len({row["task_id"] for row in finals if row.get("review_status") != "researcher_approved"})
    smoke_path = run_dir / "smoke_preflight.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else None
    snapshot_models: dict[str, set[str]] = defaultdict(set)
    snapshot_providers: dict[str, set[str]] = defaultdict(set)
    for row in finals:
        if row.get("error") is None:
            if isinstance(row.get("returned_model"), str):
                snapshot_models[row["model_arm"]].add(row["returned_model"])
            if isinstance(row.get("provider"), str):
                snapshot_providers[row["model_arm"]].add(row["provider"])
    snapshot_flags = [
        f"{arm}: multiple observed returned models or providers"
        for arm in sorted(set(snapshot_models) | set(snapshot_providers))
        if len(snapshot_models[arm]) > 1 or len(snapshot_providers[arm]) > 1
    ]
    if any(row.get("returned_model") is None for row in finals if row.get("error") is None):
        snapshot_flags.append("At least one successful response lacks a returned model ID")
    pilot_summary_path = run_dir / "summary.json"
    pilot_summary = read_json(pilot_summary_path) if pilot_summary_path.exists() else {}
    snapshot_flags.extend(str(flag) for flag in pilot_summary.get("comparability_flags", []))
    snapshot_flags = sorted(set(snapshot_flags))
    matrix_known_cost = sum(float(row.get("usage", {}).get("cost")) for row in physical
                            if isinstance(row.get("usage", {}).get("cost"), (int, float)))
    smoke_known_cost = float(smoke.get("known_cost_usd", 0)) if isinstance(smoke, dict) else 0.0
    haiku_reported_sums = [float(row["haiku_reported_probability_sum"]) for row in finals
                           if row.get("model_arm") == "haiku"
                           and row.get("probability_transform") == "haiku_relative_weights_l1"
                           and isinstance(row.get("haiku_reported_probability_sum"), (int, float))]
    base: dict[str, Any] = {
        "analysis_schema_version": 1,
        "generated_at": utc_now(),
        "status": status,
        "empirical_results_available": status == "complete",
        "labels_independently_reviewed": pending_review_count == 0,
        "paper_ready": status == "complete" and pending_review_count == 0 and not snapshot_flags,
        "run_directory": str(run_dir),
        "planned_logical_requests": len(plan),
        "final_logical_requests": len(finals),
        "physical_attempts": len(physical),
        "smoke_preflight": smoke,
        "total_physical_attempts_including_smoke": len(physical) + (smoke.get("requests_issued", 0) if isinstance(smoke, dict) else 0),
        "missing_logical_requests": len(missing),
        "missing_logical_request_examples": missing[:20],
        "failed_or_invalid_final_requests": len(errors),
        "failed_or_invalid_examples": [row.get("logical_request_id") for row in errors[:20]],
        "synthetic_rows": len(synthetic),
        "pending_review_tasks": pending_review_count,
        "actual_matrix_cost_usd_known": matrix_known_cost,
        "actual_smoke_cost_usd_known": smoke_known_cost,
        "actual_total_cost_usd_known": matrix_known_cost + smoke_known_cost,
        "haiku_weight_normalization": {
            "method": "Each complete nonnegative elicited weight is divided by the positive reported total.",
            "valid_recorded_final_responses": len(haiku_reported_sums),
            "reported_sum_min": min(haiku_reported_sums) if haiku_reported_sums else None,
            "reported_sum_max": max(haiku_reported_sums) if haiku_reported_sums else None,
            "mean_absolute_reported_sum_deviation_from_one": (
                mean(abs(value - 1.0) for value in haiku_reported_sums) if haiku_reported_sums else None
            ),
        },
        "unknown_cost_attempts": sum(not isinstance(row.get("usage", {}).get("cost"), (int, float)) for row in physical),
        "provider_snapshots": sorted({(str(row.get("model_arm")), str(row.get("returned_model")),
                                        str(row.get("provider")), str(row.get("system_fingerprint"))) for row in finals}),
        "snapshot_comparability_flags": snapshot_flags,
        "provenance_notes": pilot_summary.get("provenance_notes", []),
        "pilot_request_accounting": pilot_summary.get("request_accounting"),
        "limitations": [
            "The task labels were AI drafted and mechanically checked; independent researcher review is pending unless the ledger says otherwise.",
            "Two repeats are repeated service requests, not independent tasks.",
            "The target population is this bounded synthetic fixture and dispatch registry.",
            "Haiku uses normalized elicited relative weights, not native model probabilities; raw weights and their sums are retained in the ledger.",
        ],
    }
    operational_path = run_dir / "operational_by_split_model_condition.csv"
    _write_csv(operational_path, _operational_accounting(plan, physical, finals))
    base["operational_accounting_path"] = str(operational_path)
    tie_path = run_dir / "tie_rates_by_split_model_condition.csv"
    _write_csv(tie_path, _tie_rates(finals))
    base["tie_diagnostics_path"] = str(tie_path)
    if mixed_synthetic:
        raise ValueError("Mixed synthetic and live rows are not comparable")
    if not completed:
        return _analyze_partial(run_dir, plan, finals, base, bootstrap_replicates)
    split_counts = Counter(row["split"] for row in finals)
    if split_counts != {"pilot": 1080, "calibration": 3240, "held_out": 6480}:
        raise ValueError(f"Unexpected split row counts: {dict(split_counts)}")
    calibration = [row for row in finals if row["split"] == "calibration" and row["condition"] == "C0"]
    thresholds: dict[str, dict[str, Any]] = {}
    for model in ("jev", "luna", "haiku"):
        model_rows = [row for row in calibration if row["model_arm"] == model]
        if len(model_rows) != 180:
            raise ValueError(f"Expected 180 C0 calibration responses per model, found {len(model_rows)} for {model}")
        thresholds[model] = {}
        policies = ("P1", "P2") if model in {"jev", "luna"} else ("P2",)
        for policy in policies:
            scores = [row["policies"][policy]["gating_score"] for row in model_rows]
            if any(score is None for score in scores):
                raise ValueError(f"Missing {model} {policy} calibration score")
            thresholds[model][policy] = _fit_threshold([float(score) for score in scores])
        thresholds[model]["P3"] = {**thresholds[model]["P2"], "shared_with": "P2 on C0"}
        thresholds[model]["P4"] = {**thresholds[model]["P2"], "shared_with": "P2 on C0"}
    write_json(run_dir / "thresholds.json", thresholds)

    held = [row for row in finals if row["split"] == "held_out"]
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for row in held:
        tasks_by_id[row["task_id"]] = row
    if len(tasks_by_id) != 180:
        raise ValueError("Held-out split must contain 180 unique task prompts")
    task_records = list(tasks_by_id.values())
    samples = _cluster_samples(task_records, bootstrap_replicates)
    family_counts = Counter((row["domain"], row["scenario_family_id"]) for row in task_records)
    baseline = {(row["model_arm"], row["task_id"], row["repeat"]): row for row in held if row["condition"] == "C0"}
    evaluation: list[dict[str, Any]] = []
    for row in held:
        for policy in (("P1", "P2", "P3", "P4") if row["model_arm"] in {"jev", "luna"} else ("P2", "P3", "P4")):
            source = _policy_source(row, policy, baseline)
            score = source.get("gating_score")
            action = source.get("action_id")
            if not source.get("available") or score is None or action is None:
                raise ValueError(f"Missing held-out {policy} policy for {row['logical_request_id']}")
            accepted = float(score) >= thresholds[row["model_arm"]][policy]["cutoff"]
            evaluation.append({
                "model_arm": row["model_arm"],
                "policy": policy,
                "condition": row["condition"],
                "task_id": row["task_id"],
                "scenario_family_id": row["scenario_family_id"],
                "domain": row["domain"],
                "repeat": row["repeat"],
                "action_id": action,
                "score": float(score),
                "accepted": accepted,
                "correct": action == row["gold_action_id"],
                "brier": _brier(row) if policy != "P4" else _brier(baseline[(row["model_arm"], row["task_id"], row["repeat"])]),
            })
    _write_csv(run_dir / "held_out_policy_rows.csv", evaluation)
    eval_index = {
        (row["model_arm"], row["policy"], row["task_id"], row["condition"], row["repeat"]): row
        for row in evaluation
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation:
        grouped[(row["model_arm"], row["policy"], row["condition"], "all")].append(row)
        grouped[(row["model_arm"], row["policy"], row["condition"], row["domain"])].append(row)
    summaries = [
        {"model_arm": key[0], "policy": key[1], "condition": key[2], "domain": key[3], **_summarize(value)}
        for key, value in sorted(grouped.items())
    ]
    _write_csv(run_dir / "held_out_metrics.csv", summaries)
    summary_map = {(row["model_arm"], row["policy"], row["condition"], row["domain"]): row for row in summaries}

    contrasts: list[dict[str, Any]] = []
    for model in ("jev", "luna", "haiku"):
        for policy in (("P1", "P2", "P3", "P4") if model in {"jev", "luna"} else ("P2", "P3", "P4")):
            for condition in CONDITIONS[1:]:
                task_values: dict[str, dict[str, float]] = {}
                for task_id in tasks_by_id:
                    paired = [
                        (eval_index[(model, policy, task_id, condition, repeat)],
                         eval_index[(model, policy, task_id, "C0", repeat)])
                        for repeat in (1, 2)
                    ]
                    task_values[task_id] = {
                        "coverage_delta": mean(float(current["accepted"]) - float(base_row["accepted"]) for current, base_row in paired),
                        "accepted_wrong_rate_delta": mean(
                            float(current["accepted"] and not current["correct"]) - float(base_row["accepted"] and not base_row["correct"])
                            for current, base_row in paired
                        ),
                        "accepted_correct_yield_delta": mean(
                            float(current["accepted"] and current["correct"]) - float(base_row["accepted"] and base_row["correct"])
                            for current, base_row in paired
                        ),
                        "unconditional_action_accuracy_delta": mean(float(current["correct"]) - float(base_row["correct"]) for current, base_row in paired),
                        "action_change_rate": mean(float(current["action_id"] != base_row["action_id"]) for current, base_row in paired),
                        "accept_defer_change_rate": mean(float(current["accepted"] != base_row["accepted"]) for current, base_row in paired),
                    }
                record: dict[str, Any] = {"model_arm": model, "policy": policy, "condition": condition}
                for metric in next(iter(task_values.values())):
                    record[metric] = _interval({task_id: value[metric] for task_id, value in task_values.items()}, samples)
                record["absolute_coverage_deviation_from_c0"] = abs(record["coverage_delta"]["estimate"])
                contrasts.append(record)
    write_json(run_dir / "paired_contrasts.json", contrasts)

    mitigation: list[dict[str, Any]] = []
    for model in ("jev", "luna", "haiku"):
        for condition in CONDITIONS[1:]:
            p2 = summary_map[(model, "P2", condition, "all")]
            p3 = summary_map[(model, "P3", condition, "all")]
            base_coverage = summary_map[(model, "P2", "C0", "all")]["coverage"]
            p2_deviation = abs(p2["coverage"] - base_coverage)
            p3_deviation = abs(p3["coverage"] - base_coverage)
            yield_diff: dict[str, float] = {}
            wrong_diff: dict[str, float] = {}
            for task_id in tasks_by_id:
                p3_rows = [eval_index[(model, "P3", task_id, condition, repeat)] for repeat in (1, 2)]
                p2_rows = [eval_index[(model, "P2", task_id, condition, repeat)] for repeat in (1, 2)]
                yield_diff[task_id] = mean(float(a["accepted"] and a["correct"]) - float(b["accepted"] and b["correct"]) for a, b in zip(p3_rows, p2_rows, strict=True))
                wrong_diff[task_id] = mean(float(a["accepted"] and not a["correct"]) - float(b["accepted"] and not b["correct"]) for a, b in zip(p3_rows, p2_rows, strict=True))
            deviation_draws: list[float] = []
            for sample in samples:
                def sampled_coverage(policy: str, cond: str) -> float:
                    return mean(_task_mean(eval_index, model, policy, task_id, cond, "accepted") for task_id in sample)
                deviation_draws.append(
                    abs(sampled_coverage("P3", condition) - sampled_coverage("P3", "C0"))
                    - abs(sampled_coverage("P2", condition) - sampled_coverage("P2", "C0"))
                )
            deviation_draws.sort()
            mitigation.append({
                "model_arm": model,
                "condition": condition,
                "P2_absolute_coverage_deviation": p2_deviation,
                "P3_absolute_coverage_deviation": p3_deviation,
                "P3_minus_P2_absolute_coverage_deviation": {
                    "estimate": p3_deviation - p2_deviation,
                    "ci95_low": _quantile(deviation_draws, 0.025),
                    "ci95_high": _quantile(deviation_draws, 0.975),
                },
                "P3_minus_P2_accepted_correct_yield": _interval(yield_diff, samples),
                "P3_minus_P2_accepted_wrong_rate": _interval(wrong_diff, samples),
            })
    write_json(run_dir / "mitigation.json", mitigation)

    repeats: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        if key[3] != "all":
            continue
        by_task = defaultdict(dict)
        for row in rows:
            by_task[row["task_id"]][row["repeat"]] = row
        if any(set(pairs) != {1, 2} for pairs in by_task.values()):
            raise ValueError("Held-out repeat pairing is incomplete")
        repeats.append({
            "model_arm": key[0], "policy": key[1], "condition": key[2],
            "task_pairs": len(by_task),
            "action_disagreement_rate": mean(pair[1]["action_id"] != pair[2]["action_id"] for pair in by_task.values()),
            "accept_defer_disagreement_rate": mean(pair[1]["accepted"] != pair[2]["accepted"] for pair in by_task.values()),
        })
    _write_csv(run_dir / "repeat_disagreement.csv", repeats)

    curves: list[dict[str, Any]] = []
    for (model, policy, condition, domain), rows in sorted(grouped.items()):
        if domain != "all" or policy not in {"P2", "P3"}:
            continue
        for step in range(21):
            cutoff = step / 20
            accepted = [row for row in rows if row["score"] >= cutoff]
            curves.append({
                "model_arm": model, "policy": policy, "condition": condition,
                "cutoff": cutoff,
                "coverage": len(accepted) / len(rows),
                "selective_risk": (sum(not row["correct"] for row in accepted) / len(accepted)) if accepted else None,
                "accepted_wrong_rate": sum(not row["correct"] for row in accepted) / len(rows),
            })
    _write_csv(run_dir / "risk_coverage_curves.csv", curves)
    primary = next(row for row in contrasts if row["model_arm"] == "jev" and row["policy"] == "P2" and row["condition"] == "C2")
    luna_secondary = next(row for row in contrasts if row["model_arm"] == "luna" and row["policy"] == "P2" and row["condition"] == "C2")
    native_difference_by_task = {
        task_id: mean(
            (float(eval_index[("luna", "P2", task_id, "C2", repeat)]["accepted"])
             - float(eval_index[("luna", "P2", task_id, "C0", repeat)]["accepted"]))
            - (float(eval_index[("jev", "P2", task_id, "C2", repeat)]["accepted"])
               - float(eval_index[("jev", "P2", task_id, "C0", repeat)]["accepted"]))
            for repeat in (1, 2)
        )
        for task_id in tasks_by_id
    }
    figures = write_figures(run_dir, summary_map, primary["coverage_delta"], synthetic=status == "synthetic")
    base.update({
        "calibration_thresholds_path": str(run_dir / "thresholds.json"),
        "held_out_metrics_path": str(run_dir / "held_out_metrics.csv"),
        "paired_contrasts_path": str(run_dir / "paired_contrasts.json"),
        "mitigation_path": str(run_dir / "mitigation.json"),
        "risk_coverage_curves_path": str(run_dir / "risk_coverage_curves.csv"),
        "repeat_disagreement_path": str(run_dir / "repeat_disagreement.csv"),
        "held_out_policy_rows_path": str(run_dir / "held_out_policy_rows.csv"),
        "figures": figures,
        "held_out_task_count": len(tasks_by_id),
        "held_out_family_count": len(family_counts),
        "held_out_family_count_by_domain": dict(Counter(domain for domain, _ in family_counts)),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": BOOTSTRAP_SEED, "unit": "scenario family within domain"},
        "primary_endpoint": {
            "description": "Jev P2 signed automatic-coverage change, C2 minus C0, on held-out tasks",
            **primary["coverage_delta"],
            "predeclared_practical_effect_bound": PRACTICAL_COVERAGE_SHIFT,
        },
        "secondary_luna_endpoint": {
            "description": "Luna Decisions P2 signed automatic-coverage change, C2 minus C0, on held-out tasks",
            **luna_secondary["coverage_delta"],
        },
        "exploratory_luna_minus_jev_c2_coverage_shift": _interval(native_difference_by_task, samples),
        "interpretation_guard": "No conclusion about robustness or mitigation follows from extra P3 coverage alone; inspect wrong actions, correct yield, absolute coverage deviation, and C3/C4 controls.",
    })
    if status == "synthetic":
        base["note"] = "All results in this file come from the deterministic mock adapter and are plumbing checks only."
    write_json(run_dir / "study_results.json", base)
    _write_analysis_summary(run_dir, base)
    return base


def _write_analysis_summary(run_dir: Path, result: dict[str, Any]) -> None:
    """Write a concise public-facing inventory and interpretation gates."""
    status = result["status"]
    files = [
        "manifest.json", "preflight.json", "smoke_preflight.json", "smoke_preflight.jsonl",
        "study_results.json", "thresholds.json", "held_out_metrics.csv", "paired_contrasts.json",
        "mitigation.json", "repeat_disagreement.csv", "risk_coverage_curves.csv",
        "held_out_policy_rows.csv", "summary.json", "plan.jsonl", "intents.jsonl", "requests.jsonl",
        "responses.jsonl", "ledger.jsonl", "figure_coverage_wrong.svg", "figure_primary_delta.svg",
        "operational_by_split_model_condition.csv", "held_out_complete_pair_tasks.csv",
        "tie_rates_by_split_model_condition.csv",
    ]
    available = [name for name in files if (run_dir / name).exists()]
    text = [
        "# Analysis summary",
        "",
        f"Run status: **{status}**. Empirical results available: **{result['empirical_results_available']}**.",
        f"Independent label review completed: **{result['labels_independently_reviewed']}**. Paper-ready flag: **{result['paper_ready']}**.",
        "",
        "Read these files from this run directory when reviewing the analysis:",
        "",
        *[f"- `{name}`" for name in available],
        "",
        "The raw ledger and response files are audit data. The Jev primary and Luna secondary estimates are in `study_results.json`; frozen thresholds and all paired intervals are in `thresholds.json` and `paired_contrasts.json`.",
        "Do not treat synthetic mock output, incomplete runs, or unreviewed task labels as publication-ready evidence.",
        "",
    ]
    (run_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(text), encoding="utf-8", newline="\n")


def _analyze_partial(run_dir: Path, plan: list[dict[str, Any]], finals: list[dict[str, Any]],
                     base: dict[str, Any], bootstrap_replicates: int) -> dict[str, Any]:
    """Prespecified paired complete-case analysis with missing-pair bounds.

    A task contributes to a contrast only when both repeats in C0 and that
    condition are valid. Unobserved task-level signed coverage changes are
    bounded by [-1, 1] rather than silently treated as zero or abstention.
    """
    valid_rows = [row for row in finals if row.get("error") is None and row.get("distribution_valid")]
    valid: dict[tuple[str, str, str, int], dict[str, Any]] = {
        (row["model_arm"], row["task_id"], row["condition"], row["repeat"]): row
        for row in valid_rows
    }
    planned_tasks: dict[str, dict[str, Any]] = {}
    for row in plan:
        if row["split"] == "held_out":
            planned_tasks[row["task_id"]] = row
    if len(planned_tasks) != 180:
        raise ValueError("Partial analysis requires 180 planned held-out task IDs")
    thresholds: dict[str, dict[str, Any]] = {}
    calibration_accounting: dict[str, Any] = {}
    for model in ("jev", "luna", "haiku"):
        calibration_ids = {row["task_id"] for row in plan if row["split"] == "calibration" and row["model_arm"] == model}
        retained_ids = sorted(task_id for task_id in calibration_ids if all((model, task_id, "C0", repeat) in valid for repeat in (1, 2)))
        calibration_accounting[model] = {
            "planned_tasks": len(calibration_ids),
            "retained_complete_repeat_tasks": len(retained_ids),
            "excluded_tasks": len(calibration_ids) - len(retained_ids),
            "retained_requests": 2 * len(retained_ids),
        }
        if len(retained_ids) < MIN_PARTIAL_CALIBRATION_TASKS:
            base["note"] = (
                f"Fewer than {MIN_PARTIAL_CALIBRATION_TASKS} of 90 C0 calibration tasks have two valid repeats "
                "for at least one model; no threshold-based estimate is produced."
            )
            base["calibration_accounting"] = calibration_accounting
            write_json(run_dir / "study_results.json", base)
            _write_analysis_summary(run_dir, base)
            return base
        thresholds[model] = {}
        for policy in (("P1", "P2") if model in {"jev", "luna"} else ("P2",)):
            scores = [valid[(model, task_id, "C0", repeat)]["policies"][policy].get("gating_score")
                      for task_id in retained_ids for repeat in (1, 2)]
            if any(score is None for score in scores):
                raise ValueError(f"Valid {model} C0 response lacks {policy} score")
            thresholds[model][policy] = _fit_threshold([float(score) for score in scores])
            thresholds[model][policy]["retained_calibration_tasks"] = len(retained_ids)
        thresholds[model]["P3"] = {**thresholds[model]["P2"], "shared_with": "P2 on C0"}
        thresholds[model]["P4"] = {**thresholds[model]["P2"], "shared_with": "P2 on C0"}
    write_json(run_dir / "thresholds.json", thresholds)
    metric_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    primary_values: dict[str, float] = {}
    primary_tasks: list[dict[str, Any]] = []
    for model in ("jev", "luna", "haiku"):
        policies = ("P1", "P2", "P3", "P4") if model in {"jev", "luna"} else ("P2", "P3", "P4")
        for policy in policies:
            for condition in CONDITIONS[1:]:
                values: dict[str, dict[str, float]] = {}
                for task_id, task_plan in planned_tasks.items():
                    if not all((model, task_id, cond, repeat) in valid for cond in ("C0", condition) for repeat in (1, 2)):
                        continue
                    comparisons = []
                    for repeat in (1, 2):
                        current_raw = valid[(model, task_id, condition, repeat)]
                        baseline_raw = valid[(model, task_id, "C0", repeat)]
                        current = baseline_raw["policies"]["P2"] if policy == "P4" else current_raw["policies"][policy]
                        baseline = baseline_raw["policies"]["P2"] if policy == "P4" else baseline_raw["policies"][policy]
                        score_current, score_base = current.get("gating_score"), baseline.get("gating_score")
                        if score_current is None or score_base is None or current.get("action_id") is None or baseline.get("action_id") is None:
                            comparisons = []
                            break
                        cutoff = thresholds[model][policy]["cutoff"]
                        accept_current, accept_base = float(score_current) >= cutoff, float(score_base) >= cutoff
                        correct_current = current["action_id"] == current_raw["gold_action_id"]
                        correct_base = baseline["action_id"] == baseline_raw["gold_action_id"]
                        comparisons.append({
                            "coverage_current": float(accept_current), "coverage_base": float(accept_base),
                            "wrong_current": float(accept_current and not correct_current),
                            "wrong_base": float(accept_base and not correct_base),
                            "yield_current": float(accept_current and correct_current),
                            "yield_base": float(accept_base and correct_base),
                            "action_changed": float(current["action_id"] != baseline["action_id"]),
                            "accept_changed": float(accept_current != accept_base),
                        })
                    if len(comparisons) != 2:
                        continue
                    values[task_id] = {
                        "coverage_delta": mean(item["coverage_current"] - item["coverage_base"] for item in comparisons),
                        "accepted_wrong_rate_delta": mean(item["wrong_current"] - item["wrong_base"] for item in comparisons),
                        "accepted_correct_yield_delta": mean(item["yield_current"] - item["yield_base"] for item in comparisons),
                        "action_change_rate": mean(item["action_changed"] for item in comparisons),
                        "accept_defer_change_rate": mean(item["accept_changed"] for item in comparisons),
                    }
                    metric_rows.append({
                        "model_arm": model, "policy": policy, "condition": condition,
                        "task_id": task_id, "domain": task_plan["domain"],
                        "scenario_family_id": task_plan["scenario_family_id"],
                        "coverage_current": mean(item["coverage_current"] for item in comparisons),
                        "coverage_C0": mean(item["coverage_base"] for item in comparisons),
                        "accepted_wrong_rate_current": mean(item["wrong_current"] for item in comparisons),
                        "accepted_wrong_rate_C0": mean(item["wrong_base"] for item in comparisons),
                        "accepted_correct_yield_current": mean(item["yield_current"] for item in comparisons),
                        "accepted_correct_yield_C0": mean(item["yield_base"] for item in comparisons),
                    })
                record: dict[str, Any] = {
                    "model_arm": model, "policy": policy, "condition": condition,
                    "planned_held_out_tasks": 180,
                    "retained_complete_pair_tasks": len(values),
                    "excluded_tasks": 180 - len(values),
                    "retained_families": len({planned_tasks[task_id]["scenario_family_id"] for task_id in values}),
                }
                if values:
                    retained_task_records = [planned_tasks[task_id] for task_id in values]
                    samples = _cluster_samples(retained_task_records, bootstrap_replicates)
                    for metric in next(iter(values.values())):
                        by_task = {task_id: task_values[metric] for task_id, task_values in values.items()}
                        record[metric] = _interval(by_task, samples)
                        if metric.endswith("_delta"):
                            observed_sum = sum(by_task.values())
                            missing_tasks = 180 - len(by_task)
                            record[metric]["full_180_task_worst_case_low"] = (observed_sum - missing_tasks) / 180
                            record[metric]["full_180_task_worst_case_high"] = (observed_sum + missing_tasks) / 180
                    record["absolute_coverage_deviation_from_c0"] = abs(record["coverage_delta"]["estimate"])
                contrasts.append(record)
                if model == "jev" and policy == "P2" and condition == "C2":
                    primary_values = {task_id: task_values["coverage_delta"] for task_id, task_values in values.items()}
                    primary_tasks = [planned_tasks[task_id] for task_id in values]
    write_json(run_dir / "paired_contrasts.json", contrasts)
    _write_csv(run_dir / "held_out_complete_pair_tasks.csv", metric_rows)
    base["calibration_accounting"] = calibration_accounting
    base["partial_analysis_rule"] = (
        f"At least {MIN_PARTIAL_CALIBRATION_TASKS}/90 calibration tasks per model need two valid C0 repeats. "
        "Two valid C0 repeats and two valid condition repeats are required per held-out task. "
        "Bootstrap retained scenario families within domain and compute a task-weighted ratio of sums in each draw. "
        "Missing task-level signed effects are bounded by [-1,1]."
    )
    base["paired_contrasts_path"] = str(run_dir / "paired_contrasts.json")
    base["held_out_complete_pair_tasks_path"] = str(run_dir / "held_out_complete_pair_tasks.csv")
    base["calibration_thresholds_path"] = str(run_dir / "thresholds.json")
    if primary_values:
        primary = next(row for row in contrasts if row["model_arm"] == "jev" and row["policy"] == "P2" and row["condition"] == "C2")
        base["status"] = "partial"
        base["empirical_results_available"] = not bool(base["synthetic_rows"])
        base["paper_ready"] = False
        base["primary_endpoint"] = {
            "description": "Jev P2 C2 minus C0 coverage on held-out paired complete-case tasks",
            **primary["coverage_delta"],
            "retained_tasks": len(primary_values),
            "excluded_tasks": 180 - len(primary_values),
            "retained_families": len({task["scenario_family_id"] for task in primary_tasks}),
            "predeclared_practical_effect_bound": PRACTICAL_COVERAGE_SHIFT,
        }
        luna = next((row for row in contrasts if row["model_arm"] == "luna" and row["policy"] == "P2" and row["condition"] == "C2"), None)
        if luna and "coverage_delta" in luna:
            base["secondary_luna_endpoint"] = {
                "description": "Luna Decisions P2 C2 minus C0 coverage on held-out paired complete-case tasks",
                **luna["coverage_delta"],
                "retained_tasks": luna["retained_complete_pair_tasks"],
                "excluded_tasks": luna["excluded_tasks"],
                "retained_families": luna["retained_families"],
            }
        base["note"] = "Paired complete-case estimates are available, with worst-case full-180-task bounds for missing pairs. They may be biased if missingness depends on condition or outcome."
    else:
        base["note"] = "No complete held-out Jev C0/C2 task pairs; primary endpoint cannot be estimated."
    write_json(run_dir / "study_results.json", base)
    _write_analysis_summary(run_dir, base)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    result = analyze_study(args.run, bootstrap_replicates=args.bootstrap_replicates)
    print(f"{result['status']}: {args.run / 'study_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
