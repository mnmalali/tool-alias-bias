"""Six-call credentialed compatibility smoke before the paid full matrix."""

from __future__ import annotations

import math
import hashlib
import time
from pathlib import Path
from typing import Any

from pilot.adapters import BaseAdapter, HAIKU_PROVIDER_ONLY, make_adapters, returned_model_is_compatible
from pilot.common import append_jsonl_fsync, read_json, read_jsonl, sha256_json, utc_now, write_json
from pilot.registry import build_catalog, load_registry
from pilot.verify import ValidationError


ESTIMATED_CALL_COST = {
    "jev": 2000 * 0.042 / 1_000_000,
    "luna": 2000 * 0.10 / 1_000_000,
    "haiku": 2000 * 0.10 / 1_000_000 + 300 * 0.50 / 1_000_000,
}
SMOKE_ESTIMATED_COST_USD = 2 * sum(ESTIMATED_CALL_COST.values())


def run_smoke(run_dir: Path, dataset: dict[str, Any], *, timeout_seconds: float,
              max_cost_usd: float, resume: bool = False, max_transport_retries: int = 2,
              adapters: list[BaseAdapter] | None = None) -> dict[str, Any]:
    if max_transport_retries < 0:
        raise ValidationError("Smoke transport retry count must be nonnegative")
    summary_path = run_dir / "smoke_preflight.json"
    ledger_path = run_dir / "smoke_preflight.jsonl"
    intent_path = run_dir / "smoke_intents.jsonl"
    dataset_hash = sha256_json(dataset)
    registry = load_registry()
    task = next((task for task in dataset["tasks"] if task["task_id"] == "RI-01"), None)
    if task is None:
        raise ValidationError("Smoke task RI-01 is missing")
    adapters = adapters or make_adapters("openrouter", ["jev", "luna", "haiku"], timeout_seconds, 0.0)
    by_arm = {adapter.model_arm: adapter for adapter in adapters}
    if set(by_arm) != {"jev", "luna", "haiku"}:
        raise ValidationError("Smoke requires Jev, Luna, and Haiku adapters")
    import pilot.adapters as adapters_module
    adapter_source_hash = hashlib.sha256(Path(adapters_module.__file__).read_bytes()).hexdigest()
    identity = sha256_json({
        "dataset_hash": dataset_hash,
        "registry_hash": sha256_json(registry),
        "adapter_source_hash": adapter_source_hash,
        "timeout_seconds": timeout_seconds,
        "max_transport_retries": max_transport_retries,
        "requested_models": {arm: by_arm[arm].requested_model for arm in sorted(by_arm)},
        "catalog_hashes": {condition: build_catalog(task, condition, registry).fingerprint() for condition in ("C0", "C2")},
        "requested_haiku_provider_only": HAIKU_PROVIDER_ONLY,
    })
    if summary_path.exists():
        existing = read_json(summary_path)
        logged = read_jsonl(ledger_path)
        intents = read_jsonl(intent_path)
        if (existing.get("status") == "passed" and existing.get("identity_hash") == identity
                and len(logged) == existing.get("requests_issued") == len(intents)
                and {row.get("attempt_id") for row in logged} == {row.get("attempt_id") for row in intents}
                and len({row.get("logical_probe_id") for row in logged if row.get("passed")}) == 6):
            return existing
        raise ValidationError("Existing smoke preflight is failed, incomplete, or differs from current task/adapter settings; start a new run directory")
    if ledger_path.exists() or intent_path.exists():
        raise ValidationError("Smoke intents or responses exist without a passed summary; a call may have been billed. Reconcile before starting a new run directory")
    if resume:
        raise ValidationError("Cannot resume a live study without its passed smoke preflight")
    if max_cost_usd <= SMOKE_ESTIMATED_COST_USD * (max_transport_retries + 1):
        raise ValidationError("Cost guard is too small for six compatibility probes and their transport retry reserve")
    run_dir.mkdir(parents=True, exist_ok=True)
    known_cost = 0.0
    unpriced_reserve = 0.0
    unpriced_count = 0
    records: list[dict[str, Any]] = []
    # The 13-option Haiku catalog has been the recurring compatibility failure.
    # Test it first so a failure does not repeat five already-passing paid calls.
    sequence = [("haiku", "C2"), ("haiku", "C0"),
                ("jev", "C0"), ("jev", "C2"), ("luna", "C0"), ("luna", "C2")]
    for index, (arm, condition) in enumerate(sequence):
        catalog = build_catalog(task, condition, registry)
        logical_probe_id = f"smoke__{arm}__{condition}"
        for transport_attempt in range(max_transport_retries + 1):
            remaining_estimate = (
                ESTIMATED_CALL_COST[arm] * (max_transport_retries + 1 - transport_attempt)
                + sum(ESTIMATED_CALL_COST[future_arm] * (max_transport_retries + 1)
                      for future_arm, _ in sequence[index + 1:])
            )
            if known_cost + unpriced_reserve + remaining_estimate > max_cost_usd:
                raise ValidationError("Smoke cost reservation would exceed the user-selected local cap")
            attempt_id = f"{logical_probe_id}__a{transport_attempt}"
            append_jsonl_fsync(intent_path, {
                "attempt_id": attempt_id, "logical_probe_id": logical_probe_id,
                "transport_attempt": transport_attempt, "issued_at": utc_now(),
                "task_id": task["task_id"], "model_arm": arm, "condition": condition,
                "catalog_hash": catalog.fingerprint(),
                "status": "possibly_issued_until_smoke_record_written",
            })
            response = by_arm[arm].decide(task, catalog)
            cost = response.usage.get("cost") if isinstance(response.usage, dict) else None
            valid_cost = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(float(cost)) and float(cost) >= 0
            if valid_cost:
                known_cost += float(cost)
            else:
                # A timeout may have been billed; retain a planning reserve.
                unpriced_reserve += ESTIMATED_CALL_COST[arm]
                unpriced_count += 1
            valid = bool(
                response.error is None
                and response.distribution_valid
                and response.selected_option in catalog.option_ids
                and returned_model_is_compatible(response.requested_model, response.returned_model)
                and (arm != "haiku" or response.synthetic or response.request_payload.get("provider", {}).get("only") == [HAIKU_PROVIDER_ONLY])
                and (cost is None or valid_cost)
            )
            retryable = bool(
                not valid and isinstance(response.error, dict)
                and response.error.get("retryable") is True
                and transport_attempt < max_transport_retries
                and known_cost + unpriced_reserve <= max_cost_usd
            )
            record = {
                "attempt_id": attempt_id,
                "logical_probe_id": logical_probe_id,
                "transport_attempt": transport_attempt,
                "is_final_attempt": not retryable,
                "recorded_at": utc_now(),
                "task_id": task["task_id"],
                "condition": condition,
                "model_arm": arm,
                "requested_model": response.requested_model,
                "returned_model": response.returned_model,
                "provider": response.provider,
                "request_id": response.request_id,
                "response_status_code": response.response_status_code,
                "request_payload": response.request_payload,
                "wire_request_body": response.wire_request_body,
                "wire_request_hash": response.wire_request_hash,
                "raw_response": response.raw_response,
                "response_headers": response.response_headers,
                "selected_option": response.selected_option,
                "probabilities": response.probabilities,
                "haiku_reported_probabilities": response.reported_probabilities,
                "haiku_reported_probability_sum": response.reported_probability_sum,
                "probability_transform": response.probability_transform,
                "confidence": response.confidence,
                "distribution_valid": response.distribution_valid,
                "validation_errors": response.validation_errors,
                "usage": response.usage,
                "cost_known": valid_cost,
                "cost_reserve_usd": 0.0 if valid_cost else ESTIMATED_CALL_COST[arm],
                "latency_ms": response.latency_ms,
                "error": response.error,
                "passed": valid,
            }
            append_jsonl_fsync(ledger_path, record, preserve_key_order=True)
            records.append(record)
            if known_cost + unpriced_reserve > max_cost_usd or (not valid and not retryable):
                summary = {
                    "status": "failed",
                    "dataset_hash": dataset_hash,
                    "identity_hash": identity,
                    "generated_at": utc_now(),
                    "requests_issued": len(records),
                    "logical_probes_passed": len({row["logical_probe_id"] for row in records if row["passed"]}),
                    "max_transport_retries": max_transport_retries,
                    "known_cost_usd": known_cost,
                    "unpriced_cost_reserve_usd": unpriced_reserve,
                    "budget_accounted_cost_usd": known_cost + unpriced_reserve,
                    "unpriced_attempt_count": unpriced_count,
                    "failed_model_arm": arm,
                    "failed_condition": condition,
                    "reason": ("Smoke cost guard exceeded" if known_cost + unpriced_reserve > max_cost_usd else None)
                              or (response.error.get("message") if isinstance(response.error, dict) else None)
                              or "; ".join(response.validation_errors)
                              or "Model route, choice, returned model, or usage cost failed validation",
                    "ledger_path": str(ledger_path),
                }
                write_json(summary_path, summary)
                raise ValidationError(f"Live compatibility smoke failed on {arm} {condition}: {summary['reason']}; see {summary_path}")
            if valid:
                break
            time.sleep(min(2 ** transport_attempt, 5))
    summary = {
        "status": "passed",
        "dataset_hash": dataset_hash,
        "identity_hash": identity,
        "generated_at": utc_now(),
        "requests_issued": len(records),
        "logical_probes_passed": len(sequence),
        "transport_retry_count": len(records) - len(sequence),
        "max_transport_retries": max_transport_retries,
        "known_cost_usd": known_cost,
        "unpriced_cost_reserve_usd": unpriced_reserve,
        "budget_accounted_cost_usd": known_cost + unpriced_reserve,
        "unpriced_attempt_count": unpriced_count,
        "cost_complete": unpriced_count == 0,
        "models": {arm: by_arm[arm].requested_model for arm in by_arm},
        "requested_provider_constraints": {"haiku": {"only": [HAIKU_PROVIDER_ONLY], "allow_fallbacks": False, "require_parameters": True}},
        "observed_returned_models": {arm: sorted({row["returned_model"] for row in records if row["model_arm"] == arm and row["returned_model"] is not None}) for arm in by_arm},
        "observed_returned_providers": {arm: sorted({row["provider"] for row in records if row["model_arm"] == arm and row["provider"] is not None}) for arm in by_arm},
        "ledger_path": str(ledger_path),
        "note": "Six logical smoke probes and any separately logged physical transport retries are outside the 10,800-request study matrix and count as development expenditure.",
    }
    write_json(summary_path, summary)
    return summary
