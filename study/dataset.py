"""Combine the separately reviewed pilot and main split into one frozen input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pilot.common import PILOT_ROOT, read_json, sha256_json, write_json
from pilot.verify import ValidationError


STUDY_ROOT = Path(__file__).resolve().parent
MAIN_TASKS = STUDY_ROOT / "data" / "main_tasks.json"
COMBINED_TASKS = STUDY_ROOT / "data" / "study_tasks.json"


def combine_tasks(main_path: Path = MAIN_TASKS, output_path: Path = COMBINED_TASKS) -> dict[str, Any]:
    pilot = read_json(PILOT_ROOT / "data" / "pilot_tasks.json")
    main = read_json(main_path)
    pilot_tasks = pilot.get("tasks")
    main_tasks = main.get("tasks")
    if not isinstance(pilot_tasks, list) or len(pilot_tasks) != 30:
        raise ValidationError("The pilot source must have 30 tasks")
    if not isinstance(main_tasks, list) or len(main_tasks) != 270:
        raise ValidationError("The main source must have 270 tasks")
    combined = {
        "schema_version": 2,
        "dataset_id": "jev-tool-alias-study-300-v1",
        "status": "mechanically_verified_pending_researcher_label_review",
        "pilot_source_hash": sha256_json(pilot),
        "main_source_hash": sha256_json(main),
        "fixture": pilot["fixture"],
        "label_policy": pilot["label_policy"],
        "tasks": [*pilot_tasks, *main_tasks],
    }
    if len({task.get("task_id") for task in combined["tasks"]}) != 300:
        raise ValidationError("Combined task IDs must be unique")
    if output_path.exists() and read_json(output_path) == combined:
        return combined
    write_json(output_path, combined)
    return combined
