"""Offline checks for the public research package."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_required_paths_exist(self) -> None:
        for relative in (
            "study/run.py", "study/analysis.py", "study/data/study_tasks.json",
            "pilot/fixture-history.bundle", "pilot/data/action_registry.json",
            "results/observed-study/study-results.json", "scripts/verify_release.py",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_frozen_task_count(self) -> None:
        tasks = json.loads((ROOT / "study/data/study_tasks.json").read_text(encoding="utf-8"))["tasks"]
        self.assertEqual(len(tasks), 300)

    def test_recorded_primary_estimate(self) -> None:
        result = json.loads((ROOT / "results/observed-study/study-results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "partial")
        self.assertAlmostEqual(result["primary_endpoint"]["estimate"], -0.7780898876404494)

    def test_release_verifier(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/verify_release.py"], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
