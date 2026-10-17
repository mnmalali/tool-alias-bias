"""Restore the synthetic CopperLedger fixture and its Git history.

Run from the repository root:
    python scripts/restore_fixture.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "pilot" / "fixture-history.bundle"
TARGET = ROOT / "pilot" / "fixture_repo"
REQUIRED_TAGS = {
    "fixture/bootstrap",
    "fixture/reserve-stock-introduced",
    "fixture/price-rounding-fix",
    "fixture/webhook-retry",
    "fixture/api-health",
    "fixture/cli-json",
    "fixture/config-default",
    "fixture/reservation-regression",
    "fixture/restock-dedup",
}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not BUNDLE.is_file():
        print(f"Fixture bundle is missing: {BUNDLE}", file=sys.stderr)
        return 1
    if TARGET.exists():
        print(f"Fixture already exists: {TARGET}\nUse the existing immutable fixture; it is ignored by Git.")
        return 0

    clone = run("git", "clone", str(BUNDLE), str(TARGET))
    if clone.returncode:
        print(clone.stderr.strip(), file=sys.stderr)
        return clone.returncode
    fetch = run("git", "-C", str(TARGET), "fetch", str(BUNDLE), "refs/tags/*:refs/tags/*")
    if fetch.returncode:
        print(fetch.stderr.strip(), file=sys.stderr)
        return fetch.returncode
    tags = run("git", "-C", str(TARGET), "tag", "--list", "fixture/*")
    actual = {line for line in tags.stdout.splitlines() if line}
    missing = sorted(REQUIRED_TAGS - actual)
    if missing:
        print(f"Restored fixture is missing tags: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"Restored {TARGET.relative_to(ROOT)} with {len(actual)} fixture tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
