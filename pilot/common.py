"""Shared paths and serialization helpers for the local pilot package."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PILOT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PILOT_ROOT / "data"
FIXTURE_ROOT = PILOT_ROOT / "fixture_repo"


def utc_now() -> str:
    """Return an ISO-8601 timestamp with a UTC suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically for hashes and comparisons."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: Any, *, preserve_key_order: bool = False) -> None:
    """Append one JSON object.

    Ledger records use canonical key order for stable diffs.  Captured outbound
    requests opt into insertion-order serialization because candidate order is
    an experimental variable and must match the actual wire representation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if preserve_key_order:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        else:
            handle.write(canonical_json(value))
        handle.write("\n")


def append_jsonl_fsync(path: Path, value: Any, *, preserve_key_order: bool = False) -> None:
    """Durably journal an attempt before POST or its ledger acknowledgement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if preserve_key_order:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        else:
            handle.write(canonical_json(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object on {path}:{line_number}")
            rows.append(value)
    return rows


def short_hash(value: Any, length: int = 16) -> str:
    return sha256_json(value)[:length]


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values))
