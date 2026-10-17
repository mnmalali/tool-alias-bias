"""Safe, deterministic execution of the fixture's fixed backend actions."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import FIXTURE_ROOT, canonical_json


TIME_LINE = re.compile(r"^(Ran \d+ tests? in )\d+(?:\.\d+)?s$", re.MULTILINE)


def _fixture_tree_digest(fixture_root: Path) -> str:
    """Hash working-tree files while excluding Git metadata and interpreter cache."""
    digest = hashlib.sha256()
    for path in sorted(fixture_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(fixture_root)
        if ".git" in relative.parts or "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_output(value: str) -> str:
    """Remove only known elapsed-time noise before comparing test executions."""
    value = value.replace("\r\n", "\n")
    value = TIME_LINE.sub(r"\1<TIME>", value)
    return value.rstrip() + "\n" if value else ""


def _resolved_argv(arguments: dict[str, Any]) -> list[str]:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("run_command actions need a string argv list")
    return [sys.executable if item == "{python}" else item for item in argv]


def _search_literal(path: Path, pattern: str, fixture_root: Path) -> str:
    lines: list[str] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or candidate.suffix not in {".py", ".md", ".yml", ".yaml"}:
            continue
        text = candidate.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                lines.append(f"{candidate.relative_to(fixture_root).as_posix()}:{line_number}:{line}")
    return "\n".join(lines) + ("\n" if lines else "")


def execute_action(
    definition: dict[str, Any], timeout_seconds: float = 30.0, *, fixture_root: Path | None = None
) -> dict[str, Any]:
    """Run a fixed fixture action and capture proof-relevant behavior.

    The registry has no user-controlled arguments, so this executor deliberately
    supports only the small handler set used by the dataset.
    """
    root = (fixture_root or FIXTURE_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Fixture repository is missing: {root}")
    handler = definition["handler"]
    arguments = definition["bound_arguments"]
    before = _fixture_tree_digest(root)
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    exit_code = 0
    resolved_argv: list[str] | None = None
    try:
        if handler == "read_file":
            relative = Path(arguments["path"])
            target = (root / relative).resolve()
            if root not in target.parents and target != root:
                raise ValueError("read_file path escapes fixture")
            stdout = target.read_text(encoding="utf-8")
            if stdout and not stdout.endswith("\n"):
                stdout += "\n"
        elif handler == "search_literal":
            relative = Path(arguments["path"])
            target = (root / relative).resolve()
            if root not in target.parents and target != root:
                raise ValueError("search path escapes fixture")
            stdout = _search_literal(target, str(arguments["pattern"]), root)
        elif handler == "request_clarification":
            stdout = canonical_json(
                {
                    "status": "clarification_required",
                    "required_fields": list(arguments["required_fields"]),
                }
            ) + "\n"
        elif handler == "run_command":
            resolved_argv = _resolved_argv(arguments)
            environment = os.environ.copy()
            environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "GIT_PAGER": "cat"})
            completed = subprocess.run(
                resolved_argv,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
        else:
            raise ValueError(f"Unsupported handler: {handler}")
    except subprocess.TimeoutExpired as error:
        exit_code = 124
        stdout = error.stdout or ""
        stderr = (error.stderr or "") + f"\nTimed out after {timeout_seconds}s"
    except Exception as error:  # Capture expected validator failures as action results.
        exit_code = 1
        stderr = f"{type(error).__name__}: {error}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    after = _fixture_tree_digest(root)
    return {
        "handler": handler,
        "bound_arguments": arguments,
        "resolved_argv": resolved_argv,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "normalized_stdout": normalize_output(stdout),
        "normalized_stderr": normalize_output(stderr),
        "fixture_tree_before": before,
        "fixture_tree_after": after,
        "side_effect_free": before == after,
        "elapsed_ms": elapsed_ms,
    }


def execute_catalog_option(
    option: Any, registry: dict[str, Any], timeout_seconds: float = 30.0, *, fixture_root: Path | None = None
) -> dict[str, Any]:
    """Dispatch a visible option ID through its action map, then execute the registry backend.

    This deliberately resolves by ``option.action_id`` against the canonical
    registry rather than executing the option's copied definition. It makes the
    alias check a mapping/dispatch test rather than a same-object test.
    """
    action_id = option.action_id
    actions = registry.get("actions")
    if not isinstance(actions, dict) or action_id not in actions:
        raise KeyError(f"Visible option {option.option_id} maps to absent action {action_id}")
    result = execute_action(actions[action_id], timeout_seconds=timeout_seconds, fixture_root=fixture_root)
    result["dispatched_option_id"] = option.option_id
    result["dispatched_action_id"] = action_id
    return result


def execution_fingerprint(result: dict[str, Any]) -> str:
    """Fingerprint observable behavior while excluding expected timing variance."""
    observable = {
        "exit_code": result["exit_code"],
        "stdout": result["normalized_stdout"],
        "stderr": result["normalized_stderr"],
        "tree_before": result["fixture_tree_before"],
        "tree_after": result["fixture_tree_after"],
        "side_effect_free": result["side_effect_free"],
    }
    return hashlib.sha256(canonical_json(observable).encode("utf-8")).hexdigest()
