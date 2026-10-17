"""Mechanical validation for labels, catalog transformations, and aliases."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .common import DATA_DIR, FIXTURE_ROOT, PILOT_ROOT, sha256_json, utc_now, write_json
from .executor import execute_action, execute_catalog_option, execution_fingerprint
from .registry import CONDITIONS, PROOF_FIELDS, added_description_characters, build_catalog, load_dataset, load_registry


class ValidationError(RuntimeError):
    """A dataset issue that must block a paid run."""


def _combined_output(result: dict[str, Any]) -> str:
    return f"{result['stdout']}\n{result['stderr']}"


def _matches(result: dict[str, Any], evidence: dict[str, Any]) -> bool:
    text = _combined_output(result)
    if evidence["match_kind"] == "literal":
        return evidence["marker"] in text
    if evidence["match_kind"] == "regex":
        return re.search(evidence["marker"], text) is not None
    if evidence["match_kind"] == "all_literal":
        return all(marker in text for marker in evidence["markers"])
    if evidence["match_kind"] == "all_regex":
        return all(re.search(marker, text) is not None for marker in evidence["markers"])
    raise ValidationError(f"Unsupported evidence matcher: {evidence['match_kind']}")


def _compact_execution(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": result["exit_code"],
        "side_effect_free": result["side_effect_free"],
        "elapsed_ms": result["elapsed_ms"],
        "fingerprint": execution_fingerprint(result),
        "stdout_characters": len(result["stdout"]),
        "stderr_characters": len(result["stderr"]),
    }


def validate_dataset_structure(dataset: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = dataset.get("tasks")
    if not isinstance(tasks, list) or len(tasks) not in {30, 300}:
        raise ValidationError("Dataset must contain exactly 30 pilot or 300 study tasks")
    study_mode = len(tasks) == 300
    split_counts = Counter(task.get("split") for task in tasks)
    expected_split_counts = {"pilot": 30, "calibration": 90, "held_out": 180} if study_mode else {"pilot": 30}
    if split_counts != expected_split_counts:
        raise ValidationError(f"Split allocation must be {expected_split_counts}, found {dict(split_counts)}")
    counts = Counter(task.get("domain") for task in tasks)
    expected_domains = {"repository_inspection": len(tasks) // 3, "test_selection": len(tasks) // 3, "version_history": len(tasks) // 3}
    if counts != expected_domains:
        raise ValidationError(f"Domain allocation must be equal thirds, found {dict(counts)}")
    IDs = [task.get("task_id") for task in tasks]
    if len(IDs) != len(set(IDs)) or any(not isinstance(value, str) for value in IDs):
        raise ValidationError("Task IDs must be unique strings")
    families: dict[str, set[str]] = {}
    for task in tasks:
        family = task.get("scenario_family_id")
        if not isinstance(family, str) or not family.strip():
            raise ValidationError(f"{task.get('task_id')} lacks a scenario-family ID")
        families.setdefault(family, set()).add(str(task.get("split")))
    if any(len(splits) != 1 for splits in families.values()):
        raise ValidationError("A scenario family may not cross pilot, calibration, and held-out splits")
    if not study_mode and len(families) != len(tasks):
        raise ValidationError("Pilot scenario-family IDs must be unique")
    for split, expected_per_domain in ({"pilot": 10, "calibration": 30, "held_out": 60} if study_mode else {"pilot": 10}).items():
        split_domains = Counter(task.get("domain") for task in tasks if task.get("split") == split)
        if split_domains != {domain: expected_per_domain for domain in expected_domains}:
            raise ValidationError(f"{split} domain allocation is invalid: {dict(split_domains)}")

    required_fields = {
        "task_id", "split", "domain", "scenario_family_id", "task_text", "repository_context", "gold_action_id",
        "wrong_competitor_action_id", "label_rationale", "label_evidence", "review_status",
    }
    catalog_reports: list[dict[str, Any]] = []
    for task in tasks:
        missing = required_fields - set(task)
        if missing:
            raise ValidationError(f"{task.get('task_id')} lacks fields: {sorted(missing)}")
        if task["split"] not in expected_split_counts:
            raise ValidationError(f"{task['task_id']} has an unknown split")
        if not task["task_text"].strip() or not task["repository_context"].strip():
            raise ValidationError(f"{task['task_id']} has empty task stimulus")
        if task["domain"] not in registry["domains"]:
            raise ValidationError(f"{task['task_id']} has unknown domain")
        canonical_ids = registry["domains"][task["domain"]]["canonical_action_ids"]
        if task["gold_action_id"] not in canonical_ids:
            raise ValidationError(f"{task['task_id']} gold action is not in its canonical domain menu")
        if task["wrong_competitor_action_id"] not in canonical_ids or task["wrong_competitor_action_id"] == task["gold_action_id"]:
            raise ValidationError(f"{task['task_id']} has invalid frozen wrong competitor")
        evidence = task["label_evidence"]
        matcher = evidence.get("match_kind") if isinstance(evidence, dict) else None
        if matcher in {"literal", "regex"}:
            if set(evidence) != {"match_kind", "marker"} or not isinstance(evidence["marker"], str) or not evidence["marker"]:
                raise ValidationError(f"{task['task_id']} has invalid single-marker label evidence")
        elif matcher in {"all_literal", "all_regex"}:
            markers = evidence.get("markers")
            if set(evidence) != {"match_kind", "markers"} or not isinstance(markers, list) or len(markers) < 2:
                raise ValidationError(f"{task['task_id']} has invalid multi-marker label evidence")
            if not all(isinstance(marker, str) and marker for marker in markers):
                raise ValidationError(f"{task['task_id']} has blank multi-marker label evidence")
        else:
            raise ValidationError(f"{task['task_id']} has unsupported label evidence matcher")

        needs_mutation = task["domain"] == "test_selection" and task["gold_action_id"] != "test.request_clarification"
        mutation = task.get("mutation")
        if needs_mutation:
            if not isinstance(mutation, dict) or set(mutation) != {"path", "find", "replace"}:
                raise ValidationError(f"{task['task_id']} needs an exact test-label mutation specification")
            if not all(isinstance(mutation[field], str) and mutation[field] for field in ("path", "find", "replace")):
                raise ValidationError(f"{task['task_id']} has a blank test-label mutation field")
            if mutation["find"] == mutation["replace"]:
                raise ValidationError(f"{task['task_id']} has a no-op test-label mutation")
        elif mutation is not None:
            raise ValidationError(f"{task['task_id']} unexpectedly defines a test-label mutation")
        for condition in CONDITIONS:
            catalog = build_catalog(task, condition, registry)
            gold_option_id = registry["actions"][task["gold_action_id"]]["base_option_id"]
            wrong_option_id = registry["actions"][task["wrong_competitor_action_id"]]["base_option_id"]
            catalog_reports.append(
                {
                    "task_id": task["task_id"],
                    "condition": condition,
                    "position_seed": catalog.position_seed,
                    "position_rotation": catalog.position_rotation,
                    "position_scheme": catalog.position_scheme,
                    "counterbalance_targets": catalog.counterbalance_targets,
                    "gold_canonical_position": catalog.option_ids.index(gold_option_id),
                    "wrong_canonical_position": catalog.option_ids.index(wrong_option_id),
                    "visible_option_positions": {
                        option_id: catalog.option_ids.index(option_id) for option_id in catalog.option_ids
                    },
                    "option_count": len(catalog.options),
                    "catalog_hash": catalog.fingerprint(),
                    "added_description_characters": added_description_characters(catalog),
                }
            )
    for domain in expected_domains:
        for condition in CONDITIONS:
            rotations = sorted(
                report["position_rotation"]
                for report in catalog_reports
                if dataset_task_domain(dataset, report["task_id"]) == domain and report["condition"] == condition
            )
            expected_positions = sorted(list(range(10)) * (len(tasks) // 30))
            if rotations != expected_positions:
                raise ValidationError(
                    f"{domain} {condition} does not use balanced 0–9 Latin rows: {rotations}"
                )
            domain_reports = [
                report for report in catalog_reports
                if dataset_task_domain(dataset, report["task_id"]) == domain and report["condition"] == condition
            ]
            for field in ("gold_canonical_position", "wrong_canonical_position"):
                positions = sorted(report[field] for report in domain_reports)
                if positions != expected_positions:
                    raise ValidationError(f"{domain} {condition} does not balance {field}: {positions}")
            if condition in {"C1", "C3"}:
                extra_option_ids = [registry["alias_option_ids"][0]]
            elif condition in {"C2", "C4"}:
                extra_option_ids = list(registry["alias_option_ids"])
            elif condition == "C5":
                extra_option_ids = [
                    registry["actions"][action_id]["base_option_id"]
                    for action_id in registry["domains"][domain]["control_action_ids"]
                ]
            else:
                extra_option_ids = []
            for option_id in extra_option_ids:
                positions = sorted(report["visible_option_positions"][option_id] for report in domain_reports)
                if positions != expected_positions:
                    raise ValidationError(
                        f"{domain} {condition} does not balance added option {option_id}: {positions}"
                    )
    return catalog_reports


def dataset_task_domain(dataset: dict[str, Any], task_id: str) -> str:
    """Small explicit lookup keeps counterbalance validation independent of file order."""
    for task in dataset["tasks"]:
        if task["task_id"] == task_id:
            return str(task["domain"])
    raise ValidationError(f"Catalog refers to absent task {task_id}")


def validate_fixture_state() -> dict[str, Any]:
    if not (FIXTURE_ROOT / ".git").is_dir():
        raise ValidationError("Fixture repository is missing its Git history")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=FIXTURE_ROOT, text=True, capture_output=True, check=False
    )
    if status.returncode != 0:
        raise ValidationError(f"Could not inspect fixture Git state: {status.stderr.strip()}")
    if status.stdout.strip():
        raise ValidationError("Fixture working tree is dirty; restore a validated state before an experiment")
    tags = subprocess.run(
        ["git", "tag", "--list", "fixture/*"], cwd=FIXTURE_ROOT, text=True, capture_output=True, check=False
    )
    required_tags = {
        "fixture/bootstrap", "fixture/reserve-stock-introduced", "fixture/price-rounding-fix", "fixture/webhook-retry",
        "fixture/api-health", "fixture/cli-json", "fixture/config-default", "fixture/reservation-regression",
    }
    actual_tags = {line for line in tags.stdout.splitlines() if line}
    missing = required_tags - actual_tags
    if missing:
        raise ValidationError(f"Fixture history lacks tags: {sorted(missing)}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=FIXTURE_ROOT, text=True, capture_output=True, check=False
    )
    return {"path": str(FIXTURE_ROOT), "head": head.stdout.strip(), "tags": sorted(actual_tags)}


def validate_executable_actions(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for action_id, definition in registry["actions"].items():
        result = execute_action(definition)
        if result["exit_code"] != 0:
            raise ValidationError(f"Action {action_id} failed: {result['stderr'].strip()}")
        if not result["side_effect_free"]:
            raise ValidationError(f"Action {action_id} changed the fixture tree")
        reports[action_id] = _compact_execution(result)
    return reports


def validate_labels(dataset: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}

    def invoke(action_id: str) -> dict[str, Any]:
        if action_id not in cache:
            cache[action_id] = execute_action(registry["actions"][action_id])
        return cache[action_id]

    checks: list[dict[str, Any]] = []
    for task in dataset["tasks"]:
        domain = registry["domains"][task["domain"]]
        domain_actions = domain["canonical_action_ids"]
        matches = [action_id for action_id in domain_actions if _matches(invoke(action_id), task["label_evidence"])]
        control_matches = [action_id for action_id in domain["control_action_ids"] if _matches(invoke(action_id), task["label_evidence"])]
        gold_result = invoke(task["gold_action_id"])
        if gold_result["exit_code"] != 0:
            raise ValidationError(f"Gold action for {task['task_id']} does not execute")
        if matches != [task["gold_action_id"]]:
            raise ValidationError(
                f"{task['task_id']} label evidence is not unique to its gold action: matches {matches}"
            )
        if control_matches:
            raise ValidationError(
                f"{task['task_id']} label evidence is also satisfied by C5 control action(s): {control_matches}"
            )
        checks.append(
            {
                "task_id": task["task_id"],
                "gold_action_id": task["gold_action_id"],
                "wrong_competitor_action_id": task["wrong_competitor_action_id"],
                "evidence": task["label_evidence"],
                "matching_canonical_actions": matches,
                "matching_c5_controls": control_matches,
                "gold_execution": _compact_execution(gold_result),
                "review_status": task["review_status"],
            }
        )
    return checks


def validate_test_mutations(dataset: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Causally verify that each narrow test label detects its named behavior only.

    Each mutation is applied to a disposable copy of the fixture.  The labeled
    command must fail, while every other executable test candidate in C0/C5
    remains green.  This is deliberately stronger than matching a test name.
    """
    domain = registry["domains"]["test_selection"]
    test_action_ids = [
        action_id
        for action_id in [*domain["canonical_action_ids"], *domain["control_action_ids"]]
        if registry["actions"][action_id]["handler"] == "run_command"
    ]
    checks: list[dict[str, Any]] = []
    # The same frozen behavioral mutation can support several independently
    # worded tasks. Execute it once, but retain a task-level validation row.
    mutation_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for task in dataset["tasks"]:
        if task["domain"] != "test_selection" or task["gold_action_id"] == "test.request_clarification":
            continue
        mutation = task["mutation"]
        mutation_key = sha256_json(mutation)
        if mutation_key not in mutation_cache:
            with tempfile.TemporaryDirectory(prefix="jev-pilot-mutation-") as temporary:
                clone = Path(temporary) / "fixture_repo"
                shutil.copytree(
                    FIXTURE_ROOT,
                    clone,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
                )
                target = (clone / mutation["path"]).resolve()
                if clone.resolve() not in target.parents:
                    raise ValidationError(f"{task['task_id']} mutation path escapes fixture copy")
                source = target.read_text(encoding="utf-8")
                occurrences = source.count(mutation["find"])
                if occurrences != 1:
                    raise ValidationError(
                        f"{task['task_id']} mutation target must occur exactly once, found {occurrences}: {mutation['path']}"
                    )
                target.write_text(source.replace(mutation["find"], mutation["replace"], 1), encoding="utf-8", newline="\n")
                mutation_cache[mutation_key] = {
                    action_id: execute_action(registry["actions"][action_id], fixture_root=clone)
                    for action_id in test_action_ids
                }
        outcomes = mutation_cache[mutation_key]
        gold = outcomes[task["gold_action_id"]]
        unexpected_failures = [
            action_id for action_id, result in outcomes.items()
            if action_id != task["gold_action_id"] and result["exit_code"] != 0
        ]
        side_effecting = [action_id for action_id, result in outcomes.items() if not result["side_effect_free"]]
        if gold["exit_code"] == 0:
            raise ValidationError(f"{task['task_id']} mutation did not fail its labeled test command")
        if unexpected_failures:
            raise ValidationError(
                f"{task['task_id']} mutation also failed non-gold test command(s): {unexpected_failures}"
            )
        if side_effecting:
            raise ValidationError(f"{task['task_id']} mutation validation wrote fixture copy: {side_effecting}")
        checks.append(
            {
                "task_id": task["task_id"],
                "gold_action_id": task["gold_action_id"],
                "mutation": mutation,
                "gold_exit_code": gold["exit_code"],
                "other_executable_test_actions_remain_green": [
                    action_id for action_id in test_action_ids if action_id != task["gold_action_id"]
                ],
                "outcomes": {action_id: _compact_execution(result) for action_id, result in outcomes.items()},
            }
        )
    expected_concrete = sum(
        task["domain"] == "test_selection" and task["gold_action_id"] != "test.request_clarification"
        for task in dataset["tasks"]
    )
    if len(checks) != expected_concrete:
        raise ValidationError(f"Expected {expected_concrete} concrete test-label mutations, found {len(checks)}")
    return checks


def validate_aliases(dataset: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Prove each visible alias preserves the full invocation contract and behavior."""
    checks: list[dict[str, Any]] = []
    execution_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for task in dataset["tasks"]:
        for condition in ("C1", "C2", "C3", "C4"):
            catalog = build_catalog(task, condition, registry)
            aliases = [option for option in catalog.options if option.is_alias]
            for alias in aliases:
                canonical = next(
                    option for option in catalog.options if option.action_id == alias.action_id and not option.is_alias
                )
                structural_equal = all(alias.definition[field] == canonical.definition[field] for field in PROOF_FIELDS)
                canonical_key = (canonical.option_id, canonical.action_id)
                alias_key = (alias.option_id, alias.action_id)
                if canonical_key not in execution_cache:
                    execution_cache[canonical_key] = execute_catalog_option(canonical, registry)
                if alias_key not in execution_cache:
                    execution_cache[alias_key] = execute_catalog_option(alias, registry)
                canonical_result = execution_cache[canonical_key]
                alias_result = execution_cache[alias_key]
                equivalent = (
                    structural_equal
                    and canonical_result["dispatched_action_id"] == alias.action_id
                    and alias_result["dispatched_action_id"] == alias.action_id
                    and canonical_result["side_effect_free"]
                    and alias_result["side_effect_free"]
                    and execution_fingerprint(canonical_result) == execution_fingerprint(alias_result)
                )
                if not equivalent:
                    raise ValidationError(
                        f"Alias {alias.option_id} in {task['task_id']} {condition} is not execution-equivalent"
                    )
                checks.append(
                    {
                        "task_id": task["task_id"],
                        "condition": condition,
                        "alias_option_id": alias.option_id,
                        "canonical_option_id": canonical.option_id,
                        "canonical_dispatched_action_id": canonical_result["dispatched_action_id"],
                        "alias_dispatched_action_id": alias_result["dispatched_action_id"],
                        "action_id": alias.action_id,
                        "structural_proof_fields": list(PROOF_FIELDS),
                        "structural_equal": structural_equal,
                        "canonical_execution": _compact_execution(canonical_result),
                        "alias_execution": _compact_execution(alias_result),
                        "equivalent": equivalent,
                    }
                )
    expected_alias_checks = 8 * len(dataset["tasks"])
    if len(checks) != expected_alias_checks:
        raise ValidationError(f"Expected {expected_alias_checks} alias checks, found {len(checks)}")
    return checks


def validate_all(execute_aliases: bool = True, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a report or raise ValidationError before a model request is sent."""
    dataset = dataset or load_dataset()
    registry = load_registry()
    fixture = validate_fixture_state()
    catalogs = validate_dataset_structure(dataset, registry)
    actions = validate_executable_actions(registry)
    labels = validate_labels(dataset, registry)
    test_mutations = validate_test_mutations(dataset, registry)
    aliases = validate_aliases(dataset, registry) if execute_aliases else []
    return {
        "report_schema_version": 2,
        "generated_at": utc_now(),
        "passed": True,
        "dataset_hash": sha256_json(dataset),
        "registry_hash": sha256_json(registry),
        "fixture": fixture,
        "catalogs": catalogs,
        "executable_actions": actions,
        "label_checks": labels,
        "test_mutation_checks": test_mutations,
        "alias_checks": aliases,
        "notes": [
            "Every label is mechanically checked against a unique fixture-output predicate and is inapplicable to C5 controls.",
            "Each concrete test-selection label is also verified with a single, task-specific fixture mutation.",
            "Every C1-C4 alias is structurally compared and dispatched by visible option ID to the same executable backend.",
            "This report is not a substitute for the pending human label review recorded in the dataset."
        ],
    }


def preflight_identity(report: dict[str, Any]) -> str:
    """Hash the deterministic validation evidence, excluding timestamps and timing noise.

    The ordinary report hash is intentionally not resumable because it includes
    ``generated_at`` and elapsed measurements.  This stable identity freezes
    fixture Git state plus the execution fingerprints and catalog structure
    that determine the experiment's executable stimuli.
    """
    def compact_execution(execution: dict[str, Any]) -> dict[str, Any]:
        return {
            "exit_code": execution["exit_code"],
            "side_effect_free": execution["side_effect_free"],
            "fingerprint": execution["fingerprint"],
        }

    return sha256_json(
        {
            "dataset_hash": report["dataset_hash"],
            "registry_hash": report["registry_hash"],
            "fixture": {"head": report["fixture"]["head"], "tags": report["fixture"]["tags"]},
            "catalogs": report["catalogs"],
            "executable_actions": {
                action_id: compact_execution(execution)
                for action_id, execution in sorted(report["executable_actions"].items())
            },
            "label_checks": [
                {
                    "task_id": check["task_id"],
                    "gold_action_id": check["gold_action_id"],
                    "wrong_competitor_action_id": check["wrong_competitor_action_id"],
                    "evidence": check["evidence"],
                    "matching_canonical_actions": check["matching_canonical_actions"],
                    "matching_c5_controls": check["matching_c5_controls"],
                    "gold_execution": compact_execution(check["gold_execution"]),
                }
                for check in report["label_checks"]
            ],
            "test_mutation_checks": [
                {
                    "task_id": check["task_id"],
                    "gold_action_id": check["gold_action_id"],
                    "mutation": check["mutation"],
                    "gold_exit_code": check["gold_exit_code"],
                    "other_executable_test_actions_remain_green": check["other_executable_test_actions_remain_green"],
                    "outcomes": {
                        action_id: compact_execution(execution)
                        for action_id, execution in sorted(check["outcomes"].items())
                    },
                }
                for check in report["test_mutation_checks"]
            ],
            "alias_checks": [
                {
                    "task_id": check["task_id"],
                    "condition": check["condition"],
                    "alias_option_id": check["alias_option_id"],
                    "canonical_option_id": check["canonical_option_id"],
                    "canonical_dispatched_action_id": check["canonical_dispatched_action_id"],
                    "alias_dispatched_action_id": check["alias_dispatched_action_id"],
                    "action_id": check["action_id"],
                    "structural_equal": check["structural_equal"],
                    "canonical_execution": compact_execution(check["canonical_execution"]),
                    "alias_execution": compact_execution(check["alias_execution"]),
                    "equivalent": check["equivalent"],
                }
                for check in report["alias_checks"]
            ],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PILOT_ROOT / "validation" / "verification_report.json", help="JSON report path"
    )
    parser.add_argument("--skip-alias-execution", action="store_true", help="Only for fast development checks; not valid for a paid run")
    args = parser.parse_args(argv)
    try:
        report = validate_all(execute_aliases=not args.skip_alias_execution)
    except ValidationError as error:
        failure = {"generated_at": utc_now(), "passed": False, "error": str(error)}
        write_json(args.output, failure)
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    write_json(args.output, report)
    print(
        f"Validated {len(report['label_checks'])} labels, {len(report['catalogs'])} catalogs, "
        f"and {len(report['alias_checks'])} executable aliases. Report: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
