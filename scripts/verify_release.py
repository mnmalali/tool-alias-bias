"""Verify the tracked package and, optionally, the raw-data release asset.

Usage:
    python scripts/verify_release.py
    python scripts/verify_release.py --raw-dir results/observed-study/raw
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "observed-study"
METADATA = RESULTS / "release-metadata.json"
REQUIRED_RAW_FILES = {
    "decision_changes.jsonl", "held_out_complete_pair_tasks.csv", "intents.jsonl",
    "ledger.jsonl", "manifest.json", "operational_by_split_model_condition.csv",
    "paired_contrasts.json", "plan.jsonl", "preflight.json", "requests.jsonl",
    "responses.jsonl", "smoke_intents.jsonl", "smoke_preflight.json",
    "smoke_preflight.jsonl", "study_results.json", "summary.json", "thresholds.json",
    "tie_rates_by_split_model_condition.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def check(condition: bool, label: str) -> bool:
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Extracted raw-data directory")
    args = parser.parse_args()
    metadata = json_file(METADATA)
    failures = 0

    print("Tracked package")
    failures += not check(
        sha256(ROOT / "study" / "data" / "study_tasks.json") == metadata["tasks_sha256"],
        "frozen task file matches its SHA-256",
    )
    failures += not check(
        sha256(ROOT / "pilot" / "data" / "action_registry.json") == metadata["action_registry_sha256"],
        "action registry matches its SHA-256",
    )
    bundle = ROOT / "pilot" / "fixture-history.bundle"
    heads = subprocess.run(["git", "bundle", "list-heads", str(bundle)], text=True, capture_output=True)
    failures += not check(heads.returncode == 0 and "refs/tags/fixture/" in heads.stdout,
                           "fixture-history bundle is readable and includes fixture tags")
    study_results = json_file(RESULTS / "study-results.json")
    primary = study_results.get("primary_endpoint", {})
    failures += not check(study_results.get("status") == "partial", "recorded study status is partial")
    failures += not check(abs(primary.get("estimate", 0) - (-0.7780898876404494)) < 1e-12,
                           "Jev P2 C2-C0 estimate is -0.778090")
    with (RESULTS / "condition-metrics.csv").open(encoding="utf-8", newline="") as stream:
        planned = sum(int(row["planned_logical_requests"]) for row in csv.DictReader(stream))
    failures += not check(planned == 10800, "condition metrics sum to 10,800 planned requests")

    if args.raw_dir:
        raw = args.raw_dir.resolve()
        print("\nRaw-data release asset")
        present = {path.name for path in raw.iterdir() if path.is_file()}
        missing = sorted(REQUIRED_RAW_FILES - present)
        failures += not check(not missing, f"raw archive has all {len(REQUIRED_RAW_FILES)} expected files")
        ledger = raw / "ledger.jsonl"
        if ledger.exists():
            failures += not check(sha256(ledger) == metadata["raw_ledger_sha256"],
                                   "raw ledger matches its recorded SHA-256")
            with ledger.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip())
            failures += not check(rows == 10879, "raw ledger has 10,879 physical-attempt rows")
        plan = raw / "plan.jsonl"
        if plan.exists():
            with plan.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip())
            failures += not check(rows == 10800, "raw plan has 10,800 logical-request rows")

    print("\nAll verification checks passed." if not failures else f"\n{failures} verification check(s) failed.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
