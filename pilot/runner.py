"""Run pilot or frozen three-model tool-alias tasks with an auditable ledger."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters import BaseAdapter, HAIKU_PROVIDER_ONLY, make_adapters
from .analysis import analyze_run
from .common import PILOT_ROOT, append_jsonl, append_jsonl_fsync, read_json, read_jsonl, sha256_json, utc_now, write_json
from .registry import CONDITIONS, Catalog, build_catalog, load_dataset, load_registry, shared_stimulus, stimulus_hash
from .verify import ValidationError, preflight_identity, validate_all


RUN_ORDER_SEED = 20260930
ESTIMATED_PER_CALL_USD = {
    # Frozen planning assumptions from the protocol: 2,000 input tokens each; Haiku assumes 300 output tokens.
    "jev": 2000 * 0.042 / 1_000_000,
    "luna": 2000 * 0.10 / 1_000_000,
    "haiku": 2000 * 0.10 / 1_000_000 + 300 * 0.50 / 1_000_000,
}


def _utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _argmax(values: dict[str, float], *, neutral_tie_order: dict[str, int] | None = None) -> tuple[str, float]:
    highest = max(values.values())
    tied = [key for key, value in values.items() if value == highest]
    if neutral_tie_order is None:
        # P2 keys are visible, neutral ``tool_XX`` IDs.
        return min(tied), highest
    missing = [key for key in tied if key not in neutral_tie_order]
    if missing:
        raise ValueError(f"Missing a prespecified neutral tie rank for action(s): {missing}")
    return min(tied, key=lambda key: neutral_tie_order[key]), highest


def _neutral_action_tie_order(catalog: Catalog) -> dict[str, int]:
    """Rank actions by their frozen neutral canonical tool ID, never semantic action names."""
    ranks: dict[str, int] = {}
    for option in catalog.options:
        base_option_id = option.definition["base_option_id"]
        if not isinstance(base_option_id, str) or not base_option_id.startswith("tool_"):
            raise ValueError(f"Option {option.option_id} lacks a neutral canonical base ID")
        rank = int(base_option_id.removeprefix("tool_"))
        previous = ranks.get(option.action_id)
        if previous is not None and previous != rank:
            raise ValueError(f"Action {option.action_id} has inconsistent neutral base-ID ranks")
        ranks[option.action_id] = rank
    return ranks


def _derived_policies(response: Any, catalog: Catalog) -> dict[str, Any]:
    """Derive P1–P3 from a single raw model response; P4 is intentionally not a call."""
    selected_action = catalog.option_to_action.get(response.selected_option) if response.selected_option else None
    # Do not turn an otherwise malformed native decision record into a usable P1 score.
    # ``distribution_valid`` includes validation of the selected visible option,
    # and ``selected_action`` confirms that it resolves to the canonical menu.
    valid_selected_action = bool(response.distribution_valid and selected_action is not None)
    result: dict[str, Any] = {
        "P1": {
            "available": (
                response.model_arm in {"jev", "luna"}
                and valid_selected_action
                and _number(response.confidence) is not None
            ),
            "action_id": selected_action if response.model_arm in {"jev", "luna"} and valid_selected_action else None,
            "gating_score": _number(response.confidence) if response.model_arm in {"jev", "luna"} and valid_selected_action else None,
            "note": "Native decision-model confidence only; it is not interpreted as a correctness probability.",
        },
        "P2": {"available": False, "action_id": None, "option_id": None, "gating_score": None},
        "P3": {"available": False, "action_id": None, "action_masses": None, "gating_score": None},
        "P4": {
            "available": True,
            "action_id": None,
            "source_logical_request_id": None,
            "note": "Canonical-catalog reuse is represented by the linked C0 request and is not a new inference call.",
        },
    }
    if not response.distribution_valid or not isinstance(response.probabilities, dict):
        return result
    probabilities = {option_id: float(value) for option_id, value in response.probabilities.items()}
    p2_option, p2_score = _argmax(probabilities)
    result["P2"] = {
        "available": True,
        "action_id": catalog.option_to_action[p2_option],
        "option_id": p2_option,
        "gating_score": p2_score,
        "tie_break_rule": "lowest neutral visible tool_XX ID",
    }
    masses: dict[str, float] = {}
    for option_id, probability in probabilities.items():
        action_id = catalog.option_to_action[option_id]
        masses[action_id] = masses.get(action_id, 0.0) + probability
    p3_action, p3_score = _argmax(masses, neutral_tie_order=_neutral_action_tie_order(catalog))
    result["P3"] = {
        "available": True,
        "action_id": p3_action,
        "action_masses": masses,
        "gating_score": p3_score,
        "tie_break_rule": "lowest neutral canonical base tool_XX ID",
    }
    return result


def _build_plan(
    dataset: dict[str, Any], registry: dict[str, Any], adapters: list[BaseAdapter], repeats: int, conditions: list[str]
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for task in dataset["tasks"]:
        for condition in conditions:
            catalog = build_catalog(task, condition, registry)
            for repeat in range(1, repeats + 1):
                for adapter in adapters:
                    logical_id = f"{task['task_id']}__{condition}__r{repeat}__{adapter.model_arm}"
                    plan.append(
                        {
                            "logical_request_id": logical_id,
                            "task_id": task["task_id"],
                            "split": task["split"],
                            "scenario_family_id": task["scenario_family_id"],
                            "domain": task["domain"],
                            "condition": condition,
                            "repeat": repeat,
                            "model_arm": adapter.model_arm,
                            "requested_model": adapter.requested_model,
                            "catalog_hash": catalog.fingerprint(),
                            "stimulus_hash": stimulus_hash(task, catalog),
                            "position_seed": catalog.position_seed,
                            "position_rotation": catalog.position_rotation,
                            "catalog_mapping": catalog.mapping_records(),
                        }
                    )
    random.Random(RUN_ORDER_SEED).shuffle(plan)
    for index, record in enumerate(plan, start=1):
        record["run_order"] = index
    return plan


def _estimate_cost(plan: list[dict[str, Any]]) -> float:
    return sum(ESTIMATED_PER_CALL_USD[row["model_arm"]] for row in plan)


def _estimated_row_cost(row: dict[str, Any]) -> float:
    model_arm = row.get("model_arm")
    if model_arm not in ESTIMATED_PER_CALL_USD:
        raise ValueError(f"Cannot reserve cost for unknown model arm: {model_arm!r}")
    return ESTIMATED_PER_CALL_USD[model_arm]


def _plan_hash(plan: list[dict[str, Any]]) -> str:
    return sha256_json(plan)


def _harness_source_hash() -> str:
    """Freeze the local runner/adapters/verification implementation for resume provenance."""
    digest = hashlib.sha256()
    for path in sorted(PILOT_ROOT.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _frozen_settings(
    args: argparse.Namespace,
    dataset: dict[str, Any],
    registry: dict[str, Any],
    adapters: list[BaseAdapter],
    plan: list[dict[str, Any]],
    preflight_identity_hash: str | None,
) -> dict[str, Any]:
    """Frozen fields, except a recorded monotonic increase in the local cap."""
    return {
        "adapter": args.adapter,
        "models": [
            {"model_arm": adapter.model_arm, "requested_model": adapter.requested_model} for adapter in adapters
        ],
        "conditions": args.conditions,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "timeout_seconds": args.timeout_seconds,
        "max_transport_retries": args.max_transport_retries,
        "max_cost_usd": args.max_cost_usd if args.adapter == "openrouter" else None,
        "allow_unreviewed": bool(args.allow_unreviewed),
        "dataset_hash": sha256_json(dataset),
        "registry_hash": sha256_json(registry),
        "run_order_seed": RUN_ORDER_SEED,
        "plan_hash": _plan_hash(plan),
        "harness_source_hash": _harness_source_hash(),
        "preflight_identity_hash": preflight_identity_hash,
    }


def _validate_resume(
    run_dir: Path, expected_plan: list[dict[str, Any]], expected_settings: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Refuse a resume that would mix frozen stimuli or settings in one ledger."""
    manifest_path = run_dir / "manifest.json"
    plan_path = run_dir / "plan.jsonl"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise ValueError("--resume requires an existing manifest.json and plan.jsonl in --out")
    manifest = read_json(manifest_path)
    if manifest.get("manifest_schema_version") != 2:
        raise ValueError("--resume requires a schema-v2 run manifest; start a new run directory instead")
    stored_plan = read_jsonl(plan_path)
    if _plan_hash(stored_plan) != expected_settings["plan_hash"] or stored_plan != expected_plan:
        raise ValueError("--resume refused: the stored plan differs from the current frozen task/catalog plan")
    stored_settings = manifest.get("frozen_settings")
    if not isinstance(stored_settings, dict):
        raise ValueError("--resume refused: manifest lacks frozen_settings")
    differences = {
        key: {"stored": stored_settings.get(key), "current": value}
        for key, value in expected_settings.items()
        if stored_settings.get(key) != value
        and not (
            key == "max_cost_usd"
            and isinstance(stored_settings.get(key), (int, float))
            and isinstance(value, (int, float))
            and value > stored_settings[key]
        )
    }
    if differences:
        raise ValueError(f"--resume refused: frozen settings differ: {differences}")
    if manifest.get("preflight_identity_hash") != expected_settings["preflight_identity_hash"]:
        raise ValueError("--resume refused: stable preflight evidence differs from the stored run")
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    intents = read_jsonl(run_dir / "intents.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    responses = read_jsonl(run_dir / "responses.jsonl")
    ledger_attempt_ids = {row.get("attempt_id") for row in ledger}
    intent_attempt_ids = {row.get("attempt_id") for row in intents}
    if len(intent_attempt_ids) != len(intents) or intent_attempt_ids != ledger_attempt_ids:
        raise ValueError(
            "--resume refused: a pre-dispatch intent has no durable ledger acknowledgement, or an intent is missing. "
            "A request may have been billed; reconcile the intent and provider usage before starting a new run."
        )
    if {row.get("attempt_id") for row in requests} != ledger_attempt_ids or {row.get("attempt_id") for row in responses} != ledger_attempt_ids:
        raise ValueError("--resume refused: request/response/ledger attempt IDs differ; reconcile the interrupted write")
    plan_ids = {row["logical_request_id"] for row in stored_plan}
    seen_attempt_ids: set[str] = set()
    attempts_by_logical: dict[str, list[dict[str, Any]]] = {}
    for row in ledger:
        logical_id = row.get("logical_request_id")
        attempt_id = row.get("attempt_id")
        if not isinstance(logical_id, str) or logical_id not in plan_ids or not isinstance(attempt_id, str):
            raise ValueError("--resume refused: ledger contains an unknown or malformed request identity")
        if attempt_id in seen_attempt_ids:
            raise ValueError(f"--resume refused: duplicate attempt ID in ledger: {attempt_id}")
        seen_attempt_ids.add(attempt_id)
        attempts_by_logical.setdefault(logical_id, []).append(row)
    for logical_id, rows in attempts_by_logical.items():
        attempt_numbers = sorted(row.get("transport_attempt") for row in rows)
        if attempt_numbers != list(range(len(rows))):
            raise ValueError(f"--resume refused: noncontiguous attempts for {logical_id}")
        if sum(bool(row.get("is_final_attempt", True)) for row in rows) > 1:
            raise ValueError(f"--resume refused: multiple final attempts for {logical_id}")
    return manifest, stored_plan, ledger


def _remaining_reservation(
    plan: list[dict[str, Any]], existing: list[dict[str, Any]], max_transport_retries: int
) -> float:
    attempts_by_logical: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        logical_id = str(row["logical_request_id"])
        attempts_by_logical.setdefault(logical_id, []).append(row)
    reserved = 0.0
    for plan_row in plan:
        prior = attempts_by_logical.get(plan_row["logical_request_id"], [])
        if any(bool(row.get("is_final_attempt", True)) for row in prior):
            continue
        reserved += _estimated_row_cost(plan_row) * max(0, max_transport_retries + 1 - len(prior))
    return reserved


def _cost_state(existing: list[dict[str, Any]], adapter_kind: str) -> tuple[float, float, int]:
    """Return known cost, conservative reserve for unpriced physical calls, and their count."""
    known = 0.0
    unpriced_reserve = 0.0
    unpriced_count = 0
    for row in existing:
        cost = _number(row.get("usage", {}).get("cost"))
        if cost is not None:
            known += cost
        elif adapter_kind == "openrouter":
            unpriced_reserve += _estimated_row_cost(row)
            unpriced_count += 1
    return known, unpriced_reserve, unpriced_count


def _retryable(response: Any) -> bool:
    return bool(
        isinstance(response.error, dict)
        and response.error.get("retryable") is True
    )


def _pending_reviews(dataset: dict[str, Any]) -> list[str]:
    return [task["task_id"] for task in dataset["tasks"] if task.get("review_status") != "researcher_approved"]


def _get_task(dataset: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in dataset["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def _record_from_response(
    plan_row: dict[str, Any], task: dict[str, Any], catalog: Catalog, response: Any, transport_attempt: int, is_final_attempt: bool,
    retry_reason: str | None,
) -> dict[str, Any]:
    selected_action = catalog.option_to_action.get(response.selected_option) if response.selected_option else None
    policies = _derived_policies(response, catalog)
    policies["P4"] = {
        "available": True,
        "action_id": selected_action if plan_row["condition"] == "C0" else None,
        "source_logical_request_id": (
            f"{task['task_id']}__C0__r{plan_row['repeat']}__{response.model_arm}"
        ),
        "reused_current_raw_response": plan_row["condition"] == "C0",
        "note": "Reuse this exact C0 logical request; P4 adds no model inference call.",
    }
    return {
        "attempt_id": f"{plan_row['logical_request_id']}__a{transport_attempt}",
        "logical_request_id": plan_row["logical_request_id"],
        "run_order": plan_row["run_order"],
        "transport_attempt": transport_attempt,
        "is_final_attempt": is_final_attempt,
        "retry_reason": retry_reason,
        "recorded_at": utc_now(),
        "task_id": task["task_id"],
        "split": task["split"],
        "domain": task["domain"],
        "scenario_family_id": task["scenario_family_id"],
        "condition": plan_row["condition"],
        "repeat": plan_row["repeat"],
        "model_arm": response.model_arm,
        "requested_model": response.requested_model,
        "returned_model": response.returned_model,
        "provider": response.provider,
        "provider_request_id": response.request_id,
        "response_status_code": response.response_status_code,
        "response_headers": response.response_headers,
        "system_fingerprint": response.system_fingerprint,
        "catalog_hash": catalog.fingerprint(),
        "stimulus_hash": stimulus_hash(task, catalog),
        "wire_request_hash": response.wire_request_hash,
        "position_seed": catalog.position_seed,
        "position_rotation": catalog.position_rotation,
        "catalog_mapping": catalog.mapping_records(),
        "gold_action_id": task["gold_action_id"],
        "wrong_competitor_action_id": task["wrong_competitor_action_id"],
        "label_evidence": task["label_evidence"],
        "label_rationale": task["label_rationale"],
        "review_status": task["review_status"],
        "selected_option": response.selected_option,
        "selected_action": selected_action,
        "canonical_action_correct": selected_action == task["gold_action_id"] if selected_action is not None else None,
        "option_probabilities": response.probabilities,
        "haiku_reported_probabilities": response.reported_probabilities,
        "haiku_reported_probability_sum": response.reported_probability_sum,
        "probability_transform": response.probability_transform,
        "confidence": response.confidence,
        "distribution_valid": response.distribution_valid,
        "validation_errors": response.validation_errors,
        "policies": policies,
        "latency_ms": response.latency_ms,
        "usage": response.usage,
        "error": response.error,
        "synthetic": response.synthetic,
    }


def _write_plan(path: Path, plan: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    for row in plan:
        append_jsonl(path, row)


def run_experiment(args: argparse.Namespace) -> Path:
    if args.adapter == "openrouter" and not args.allow_live:
        raise ValueError("Live mode is intentionally gated. Re-run with --allow-live after label review and API access are ready.")
    if args.adapter == "openrouter" and args.skip_preflight:
        raise ValueError("A paid run cannot skip mechanical preflight validation.")
    if args.resume and args.out is None:
        raise ValueError("--resume requires an explicit existing --out directory")
    if args.repeats != 2 or set(args.conditions) != set(CONDITIONS):
        # Development subsets are permitted, but their manifests must not be mistaken for the full pilot.
        full_matrix = False
    else:
        full_matrix = True

    dataset = load_dataset(str(args.dataset) if args.dataset else None)
    registry = load_registry()
    pending = _pending_reviews(dataset)
    if args.adapter == "openrouter" and pending and not args.allow_unreviewed:
        raise ValueError(
            "Paid run blocked: researcher label review is pending for " + ", ".join(pending) + ". "
            "Update review_status to researcher_approved after review, or deliberately use --allow-unreviewed."
        )
    adapters = make_adapters(args.adapter, args.models, args.timeout_seconds, args.temperature)
    run_dir = args.out or (PILOT_ROOT / "runs" / f"{args.adapter}-{_utc_run_id()}")
    if args.resume and not run_dir.is_dir():
        raise ValueError(f"--resume requires an existing run directory: {run_dir}")
    permitted_study_preflight_files = {"smoke_preflight.json", "smoke_preflight.jsonl", "smoke_intents.jsonl"}
    if run_dir.exists() and any(child.name not in permitted_study_preflight_files for child in run_dir.iterdir()) and not args.resume:
        raise ValueError(f"Run directory already contains data: {run_dir}. Use a new --out or --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)

    preflight_report = None
    if not args.skip_preflight:
        preflight_report = validate_all(execute_aliases=True, dataset=dataset)
        if not args.resume:
            write_json(run_dir / "preflight.json", preflight_report)

    preflight_identity_hash = preflight_identity(preflight_report) if preflight_report is not None else None
    expected_plan = _build_plan(dataset, registry, adapters, args.repeats, args.conditions)
    expected_settings = _frozen_settings(
        args, dataset, registry, adapters, expected_plan, preflight_identity_hash
    )
    if args.resume:
        manifest, plan, existing = _validate_resume(run_dir, expected_plan, expected_settings)
        previous_cap = manifest.get("max_cost_usd")
        if args.adapter == "openrouter" and isinstance(previous_cap, (int, float)) and args.max_cost_usd > previous_cap:
            manifest.setdefault("cost_cap_history", []).append({
                "changed_at": utc_now(), "previous_cap_usd": previous_cap,
                "new_cap_usd": args.max_cost_usd,
                "reason": "User-selected monotonic cap increase on resume",
            })
            manifest["max_cost_usd"] = args.max_cost_usd
            manifest["frozen_settings"]["max_cost_usd"] = args.max_cost_usd
    else:
        plan = expected_plan
        existing = []
        manifest = {
            "manifest_schema_version": 2,
            "created_at": utc_now(),
            "status": "running",
            "adapter": args.adapter,
            "synthetic": args.adapter == "mock",
            "full_30x6x2x2_matrix": full_matrix and len(args.models) == 2 and len(dataset["tasks"]) == 30,
            "full_30x6x2x3_matrix": full_matrix and set(args.models) == {"jev", "luna", "haiku"} and len(dataset["tasks"]) == 30,
            "full_300x6x2x2_matrix": full_matrix and len(args.models) == 2 and len(dataset["tasks"]) == 300,
            "full_300x6x2x3_matrix": full_matrix and set(args.models) == {"jev", "luna", "haiku"} and len(dataset["tasks"]) == 300,
            "planned_requests": len(plan),
            "planned_maximum_physical_requests": len(plan) * (args.max_transport_retries + 1),
            "logical_matrix": {
                "tasks": len(dataset["tasks"]),
                "conditions": args.conditions,
                "repeats": args.repeats,
                "models": args.models,
            },
            "run_order_seed": RUN_ORDER_SEED,
            "requested_provider_constraints": {"haiku": {"only": [HAIKU_PROVIDER_ONLY], "allow_fallbacks": False, "require_parameters": True}} if args.adapter == "openrouter" else {},
            "dataset_hash": expected_settings["dataset_hash"],
            "registry_hash": expected_settings["registry_hash"],
            "plan_hash": expected_settings["plan_hash"],
            "preflight_report_hash": sha256_json(preflight_report) if preflight_report else None,
            "preflight_identity_hash": preflight_identity_hash,
            "harness_source_hash": expected_settings["harness_source_hash"],
            "temperature": args.temperature,
            "haiku_temperature_sent": False,
            "timeout_seconds": args.timeout_seconds,
            "max_transport_retries": args.max_transport_retries,
            "max_cost_usd": args.max_cost_usd if args.adapter == "openrouter" else None,
            "cost_guard_semantics": (
                "A pre-dispatch reservation guard using frozen planning rates, not a provider-side hard billing cap; "
                "actual usage is known only after a response."
                if args.adapter == "openrouter" else None
            ),
            "estimated_logical_cost_usd": None,
            "estimated_maximum_physical_cost_usd": None,
            "cost_assumptions": {
                "jev": {"input_tokens_per_call": 2000, "input_usd_per_million": 0.042, "output_usd_per_million": 0},
                "luna": {"input_tokens_per_call": 2000, "input_usd_per_million": 0.10, "output_usd_per_million": 0},
                "haiku": {
                    "input_tokens_per_call": 2000,
                    "output_tokens_per_call": 300,
                    "input_usd_per_million": 0.10,
                    "output_usd_per_million": 0.50,
                },
            },
            "pending_label_reviews": pending,
            "allow_unreviewed": bool(args.allow_unreviewed),
            "frozen_settings": expected_settings,
            "notes": [
                "Raw requests and responses are saved separately without API keys.",
                "An fsynced intent precedes each adapter call; resume refuses dangling intents. A process crash can leave billing unknown even with an intent.",
                "Only explicitly transient transport/rate errors can retry; every physical attempt is ledgered.",
                "Known cost plus a conservative reserve for unpriced and outstanding attempts is checked before dispatch; this is not a provider-side hard billing cap.",
                "P4 links to the paired C0 logical request and has no separate request count.",
            ],
        }

    estimated_logical_cost = _estimate_cost(plan)
    estimated_maximum_physical_cost = estimated_logical_cost * (args.max_transport_retries + 1)
    if args.adapter == "openrouter" and estimated_maximum_physical_cost > args.max_cost_usd + 1e-12:
        raise ValueError(
            f"Worst-case physical-request estimate ${estimated_maximum_physical_cost:.5f} exceeds "
            f"--max-cost-usd ${args.max_cost_usd:.5f}; no request was sent."
        )
    if args.adapter == "openrouter":
        manifest["estimated_logical_cost_usd"] = round(estimated_logical_cost, 10)
        manifest["estimated_maximum_physical_cost_usd"] = round(estimated_maximum_physical_cost, 10)
    if not args.resume:
        _write_plan(run_dir / "plan.jsonl", plan)
    write_json(run_dir / "manifest.json", manifest)

    completed_logical_ids = {
        row.get("logical_request_id") for row in existing if row.get("is_final_attempt", True) and row.get("logical_request_id")
    }
    adapters_by_arm = {adapter.model_arm: adapter for adapter in adapters}
    known_actual_cost, unpriced_attempt_reserve, unpriced_attempt_count = _cost_state(existing, args.adapter)
    remaining_reservation = _remaining_reservation(plan, existing, args.max_transport_retries)
    if args.adapter == "openrouter" and (
        known_actual_cost + unpriced_attempt_reserve + remaining_reservation > args.max_cost_usd + 1e-12
    ):
        raise ValueError(
            "Known cost plus conservative reservation for unpriced/outstanding attempts exceeds the cap; "
            "no further request was sent."
        )
    stopped_reason: str | None = None
    physical_requests = len(existing)
    consecutive_invalid_by_arm = {arm: 0 for arm in args.models}
    attempts_by_logical: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        attempts_by_logical.setdefault(str(row["logical_request_id"]), []).append(row)

    for plan_row in plan:
        if plan_row["logical_request_id"] in completed_logical_ids:
            continue
        prior_attempts = attempts_by_logical.get(plan_row["logical_request_id"], [])
        first_attempt = len(prior_attempts)
        if first_attempt > args.max_transport_retries:
            raise ValueError(f"No permitted retry remains for unfinished logical request {plan_row['logical_request_id']}")
        task = _get_task(dataset, plan_row["task_id"])
        catalog = build_catalog(task, plan_row["condition"], registry)
        adapter = adapters_by_arm[plan_row["model_arm"]]
        retry_reason = (
            prior_attempts[-1].get("error", {}).get("type")
            if prior_attempts and isinstance(prior_attempts[-1].get("error"), dict)
            else None
        )
        for transport_attempt in range(first_attempt, args.max_transport_retries + 1):
            remaining_reservation = _remaining_reservation(plan, existing, args.max_transport_retries)
            if args.adapter == "openrouter" and (
                known_actual_cost + unpriced_attempt_reserve + remaining_reservation > args.max_cost_usd + 1e-12
            ):
                stopped_reason = "cost_cap_reservation_would_be_exceeded"
                break
            issued_at = utc_now()
            attempt_id = f"{plan_row['logical_request_id']}__a{transport_attempt}"
            append_jsonl_fsync(run_dir / "intents.jsonl", {
                "attempt_id": attempt_id,
                "logical_request_id": plan_row["logical_request_id"],
                "issued_at": issued_at,
                "model_arm": plan_row["model_arm"],
                "condition": plan_row["condition"],
                "task_id": plan_row["task_id"],
                "stimulus_hash": plan_row["stimulus_hash"],
                "status": "possibly_issued_until_ledger_acknowledged",
            })
            response = adapter.decide(task, catalog)
            retriable = _retryable(response) and transport_attempt < args.max_transport_retries
            record = _record_from_response(
                plan_row, task, catalog, response, transport_attempt, not retriable, retry_reason
            )
            append_jsonl(
                run_dir / "requests.jsonl",
                {
                    "attempt_id": record["attempt_id"],
                    "issued_at": issued_at,
                    "logical_request_id": record["logical_request_id"],
                    "request_payload": response.request_payload,
                    "wire_request_body": response.wire_request_body,
                    "wire_request_hash": response.wire_request_hash,
                    "stimulus": shared_stimulus(task, catalog),
                },
                preserve_key_order=True,
            )
            append_jsonl(
                run_dir / "responses.jsonl",
                {
                    "attempt_id": record["attempt_id"],
                    "received_at": utc_now(),
                    "raw_response": response.raw_response,
                    "response_status_code": response.response_status_code,
                    "response_headers": response.response_headers,
                    "error": response.error,
                    "usage": response.usage,
                },
            )
            append_jsonl_fsync(run_dir / "ledger.jsonl", record)
            existing.append(record)
            attempts_by_logical.setdefault(plan_row["logical_request_id"], []).append(record)
            physical_requests += 1
            cost = _number(response.usage.get("cost"))
            if cost is not None:
                known_actual_cost += cost
            elif args.adapter == "openrouter":
                unpriced_attempt_reserve += _estimated_row_cost(plan_row)
                unpriced_attempt_count += 1
            if known_actual_cost + unpriced_attempt_reserve > (args.max_cost_usd if args.adapter == "openrouter" else math.inf) + 1e-12:
                stopped_reason = "actual_or_reserved_cost_cap_exceeded"
                break
            if retriable:
                retry_reason = response.error.get("type") if isinstance(response.error, dict) else "transport_error"
                time.sleep(min(args.backoff_seconds * (2 ** transport_attempt), 5.0))
                continue
            if response.error is not None or not response.distribution_valid:
                consecutive_invalid_by_arm[response.model_arm] += 1
                status = response.response_status_code
                if isinstance(status, int) and status in {400, 401, 402, 403, 404, 413, 422}:
                    stopped_reason = f"fatal_http_{status}_{response.model_arm}"
                elif consecutive_invalid_by_arm[response.model_arm] >= 3:
                    stopped_reason = f"three_consecutive_invalid_{response.model_arm}"
            else:
                consecutive_invalid_by_arm[response.model_arm] = 0
            break
        if stopped_reason:
            break

    manifest.update(
        {
            "status": "completed" if stopped_reason is None else "stopped",
            "completed_at": utc_now(),
            "stopped_reason": stopped_reason,
            "physical_requests_written": physical_requests,
            "known_actual_cost_usd": round(known_actual_cost, 10),
            "unpriced_attempt_reserve_usd": round(unpriced_attempt_reserve, 10),
            "unpriced_attempt_count": unpriced_attempt_count,
            "cost_complete": unpriced_attempt_count == 0,
            "budget_accounted_cost_usd": round(known_actual_cost + unpriced_attempt_reserve, 10),
            "last_resumed_at": utc_now() if args.resume else None,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    analyze_run(run_dir)
    return run_dir


def _parse_conditions(value: str) -> list[str]:
    conditions = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in CONDITIONS]
    if invalid or not conditions:
        raise argparse.ArgumentTypeError(f"Conditions must be a nonempty comma list drawn from {', '.join(CONDITIONS)}")
    if len(conditions) != len(set(conditions)):
        raise argparse.ArgumentTypeError("Conditions must not repeat")
    return conditions


def _parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [model for model in models if model not in {"jev", "luna", "haiku"}]
    if invalid or not models:
        raise argparse.ArgumentTypeError("Models must be a nonempty comma list drawn from jev,luna,haiku")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("Models must not repeat")
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("mock", "openrouter"), default="mock")
    parser.add_argument("--allow-live", action="store_true", help="Required acknowledgement before any OpenRouter request")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Deliberately bypass the human label-review gate for live use")
    parser.add_argument("--out", type=Path, help="Directory for this run's JSONL ledgers and summary")
    parser.add_argument("--dataset", type=Path, help="Optional frozen 300-task JSON dataset; defaults to the 30-task pilot")
    parser.add_argument("--resume", action="store_true", help="Skip logical requests already finalized in an existing run directory")
    parser.add_argument("--models", type=_parse_models, default=["jev", "luna", "haiku"])
    parser.add_argument("--conditions", type=_parse_conditions, default=list(CONDITIONS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.50,
        help="Finite pre-dispatch planning-reservation guard, not a provider-side hard billing cap",
    )
    parser.add_argument("--max-transport-retries", type=int, default=0)
    parser.add_argument("--backoff-seconds", type=float, default=0.5)
    parser.add_argument("--skip-preflight", action="store_true", help="Allowed only for local mock development")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    numeric_arguments = {
        "--temperature": args.temperature,
        "--timeout-seconds": args.timeout_seconds,
        "--max-cost-usd": args.max_cost_usd,
        "--backoff-seconds": args.backoff_seconds,
    }
    if any(not math.isfinite(value) for value in numeric_arguments.values()):
        parser.error("Temperature, timeout, cost guard, and backoff must be finite numbers")
    if (
        args.temperature < 0
        or args.timeout_seconds <= 0
        or args.max_cost_usd <= 0
        or args.max_transport_retries < 0
        or args.backoff_seconds < 0
    ):
        parser.error("Temperature and retry backoff must be nonnegative; timeout and cost guard must be positive")
    try:
        run_dir = run_experiment(args)
    except (ValueError, ValidationError) as error:
        print(f"Pilot runner stopped before or during execution: {error}", file=sys.stderr)
        return 2
    status = read_json(run_dir / "manifest.json").get("status", "unknown")
    print(f"Run {status}: {run_dir}\nSummary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
