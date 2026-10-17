"""Canonical registry loading and the six frozen catalog transformations."""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass, replace
from typing import Any

from .common import DATA_DIR, canonical_json, read_json, sha256_json


CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5")
ALIAS_CONDITIONS = frozenset({"C1", "C2", "C3", "C4"})
OPTION_ID_RE = re.compile(r"^tool_\d{2}$")
PROOF_FIELDS = (
    "implementation_id",
    "handler",
    "bound_arguments",
    "argument_schema",
    "permissions",
    "preconditions",
    "observable_effect",
    "description",
)


@dataclass(frozen=True)
class Option:
    """One visible option coupled to its hidden executable backend action."""

    option_id: str
    action_id: str
    is_alias: bool
    definition: dict[str, Any]
    display_description: str

    def for_model(self) -> dict[str, str]:
        return {"option_id": self.option_id, "description": self.display_description}

    def mapping_record(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "action_id": self.action_id,
            "is_alias": self.is_alias,
            "implementation_id": self.definition["implementation_id"],
            "handler": self.definition["handler"],
            "bound_arguments": copy.deepcopy(self.definition["bound_arguments"]),
            "argument_schema": copy.deepcopy(self.definition["argument_schema"]),
            "permissions": copy.deepcopy(self.definition["permissions"]),
            "preconditions": copy.deepcopy(self.definition["preconditions"]),
            "observable_effect": self.definition["observable_effect"],
            "display_description": self.display_description,
            "display_description_characters": len(self.display_description),
        }


@dataclass(frozen=True)
class Catalog:
    task_id: str
    condition: str
    position_seed: int
    position_rotation: int
    position_scheme: str
    counterbalance_targets: dict[str, int]
    options: tuple[Option, ...]

    @property
    def option_ids(self) -> list[str]:
        return [option.option_id for option in self.options]

    @property
    def option_to_action(self) -> dict[str, str]:
        return {option.option_id: option.action_id for option in self.options}

    def model_options(self) -> list[dict[str, str]]:
        return [option.for_model() for option in self.options]

    def mapping_records(self) -> list[dict[str, Any]]:
        return [option.mapping_record() for option in self.options]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "task_id": self.task_id,
                "condition": self.condition,
                "position_seed": self.position_seed,
                "position_rotation": self.position_rotation,
                "position_scheme": self.position_scheme,
                "counterbalance_targets": self.counterbalance_targets,
                "options": self.model_options(),
                "mapping": self.mapping_records(),
            }
        )


def load_registry() -> dict[str, Any]:
    return read_json(DATA_DIR / "action_registry.json")


def load_dataset(path: str | None = None) -> dict[str, Any]:
    """Load the pilot or an explicitly frozen study dataset."""
    from pathlib import Path

    return read_json(Path(path) if path is not None else DATA_DIR / "pilot_tasks.json")


def task_index(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in dataset["tasks"]}


def _seed(domain: str, condition: str, task_id: str) -> int:
    """A fixed task-specific seed for non-target option positions."""
    digest = hashlib.sha256(f"jev-tool-alias-pilot-v2|{domain}|{condition}|{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _latin_row(task_id: str) -> int:
    """Return a 0–9 row, cycling for the 90/180-task study splits."""
    match = re.search(r"(\d+)$", task_id)
    if not match:
        raise ValueError(f"Task ID lacks a numeric ordinal: {task_id}")
    ordinal = int(match.group(1))
    if ordinal < 1:
        raise ValueError(f"Task ID ordinal must be positive: {task_id}")
    return (ordinal - 1) % 10


def _counterbalance_targets(
    task: dict[str, Any], condition: str, domain: dict[str, Any], actions: dict[str, Any], alias_ids: list[str]
) -> dict[str, int]:
    """Place gold, wrong, aliases, and C5 controls in balanced neutral positions.

    Ten tasks cannot populate every one of 11/13 possible positions for a
    target class, so each class occupies positions 0–9 exactly once.  The
    targets are stable across C0–C5 for paired canonical choices and use
    disjoint Latin offsets for wrong competitors and added options.
    """
    row = _latin_row(task["task_id"])
    gold_base = actions[task["gold_action_id"]]["base_option_id"]
    wrong_base = actions[task["wrong_competitor_action_id"]]["base_option_id"]
    targets = {gold_base: row, wrong_base: (row + 3) % 10}
    if condition in {"C1", "C3"}:
        targets[alias_ids[0]] = (row + 6) % 10
    elif condition in {"C2", "C4"}:
        for offset, alias_id in enumerate(alias_ids, start=6):
            targets[alias_id] = (row + offset) % 10
    elif condition == "C5":
        for offset, action_id in enumerate(domain["control_action_ids"], start=6):
            targets[actions[action_id]["base_option_id"]] = (row + offset) % 10
    if len(targets) != len(set(targets)) or len(targets.values()) != len(set(targets.values())):
        raise ValueError(f"{task['task_id']} {condition} has colliding counterbalance targets")
    return targets


def _place_options(options: list[Option], targets: dict[str, int], seed: int) -> list[Option]:
    """Reserve target positions, then use a frozen shuffle only for filler options."""
    by_id = {option.option_id: option for option in options}
    if set(targets) - set(by_id):
        raise ValueError(f"Counterbalance targets reference absent visible option(s): {sorted(set(targets) - set(by_id))}")
    slots: list[Option | None] = [None] * len(options)
    for option_id, position in targets.items():
        if not 0 <= position < len(slots) or slots[position] is not None:
            raise ValueError(f"Invalid or colliding target position for {option_id}: {position}")
        slots[position] = by_id[option_id]
    filler = [option for option in options if option.option_id not in targets]
    random.Random(seed).shuffle(filler)
    for index, option in zip((index for index, current in enumerate(slots) if current is None), filler, strict=True):
        slots[index] = option
    if any(option is None for option in slots):
        raise ValueError("Counterbalance placement left an empty slot")
    return [option for option in slots if option is not None]


def _option(option_id: str, action_id: str, is_alias: bool, actions: dict[str, Any]) -> Option:
    if action_id not in actions:
        raise KeyError(f"Unknown action ID: {action_id}")
    # Each visible option owns a separate record. Equality is validated later;
    # alias proof must not rely on Python object identity.
    definition = copy.deepcopy(actions[action_id])
    return Option(
        option_id=option_id,
        action_id=action_id,
        is_alias=is_alias,
        definition=definition,
        display_description=definition["description"],
    )


def _normalized_display_description(description: str, target_characters: int) -> str:
    """Equalize visible description load without altering executable-action semantics.

    The short suffix states only an invariant of this fixture—each candidate is
    a fixed action—and dot padding is intentionally nonsemantic.  Its length is
    frozen per domain in the registry so C1/C3, C2/C4, and C5 add exactly one
    or three equal-sized descriptions respectively.
    """
    if len(description) > target_characters:
        raise ValueError(f"Description exceeds frozen target length ({len(description)} > {target_characters})")
    if len(description) == target_characters:
        return description
    prefix = " [fixed action:"
    suffix = "]"
    filler_length = target_characters - len(description) - len(prefix) - len(suffix)
    if filler_length < 0:
        raise ValueError("Frozen description target leaves no space for a neutral padding suffix")
    return f"{description}{prefix}{'.' * filler_length}{suffix}"


def _normalize_option_descriptions(options: list[Option], target_characters: int) -> list[Option]:
    return [
        replace(option, display_description=_normalized_display_description(option.definition["description"], target_characters))
        for option in options
    ]


def build_catalog(task: dict[str, Any], condition: str, registry: dict[str, Any] | None = None) -> Catalog:
    """Apply C0–C5 mechanically to one task's canonical domain menu."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    registry = registry or load_registry()
    actions = registry["actions"]
    domain = registry["domains"][task["domain"]]
    target_characters = domain.get("prompt_description_target_characters")
    if not isinstance(target_characters, int) or target_characters <= 0:
        raise ValueError(f"{task['domain']} lacks a positive frozen prompt-description target")
    canonical_ids = domain["canonical_action_ids"]
    options = [_option(actions[action_id]["base_option_id"], action_id, False, actions) for action_id in canonical_ids]
    alias_ids = registry["alias_option_ids"]
    gold = task["gold_action_id"]
    wrong = task["wrong_competitor_action_id"]

    if condition == "C1":
        options.append(_option(alias_ids[0], gold, True, actions))
    elif condition == "C2":
        options.extend(_option(alias_id, gold, True, actions) for alias_id in alias_ids)
    elif condition == "C3":
        options.append(_option(alias_ids[0], wrong, True, actions))
    elif condition == "C4":
        options.extend(_option(alias_id, wrong, True, actions) for alias_id in alias_ids)
    elif condition == "C5":
        options.extend(_option(actions[action_id]["base_option_id"], action_id, False, actions) for action_id in domain["control_action_ids"])

    options = _normalize_option_descriptions(options, target_characters)

    seed = _seed(task["domain"], condition, task["task_id"])
    rotation = _latin_row(task["task_id"])
    targets = _counterbalance_targets(task, condition, domain, actions, alias_ids)
    options = _place_options(options, targets, seed)
    catalog = Catalog(
        task_id=task["task_id"],
        condition=condition,
        position_seed=seed,
        position_rotation=rotation,
        position_scheme="target-position-latin-v2",
        counterbalance_targets=targets,
        options=tuple(options),
    )
    validate_catalog(task, catalog, registry)
    return catalog


def build_all_catalogs(dataset: dict[str, Any] | None = None, registry: dict[str, Any] | None = None) -> list[Catalog]:
    dataset = dataset or load_dataset()
    registry = registry or load_registry()
    return [build_catalog(task, condition, registry) for task in dataset["tasks"] for condition in CONDITIONS]


def _proof_signature(option: Option) -> str:
    return canonical_json({field: option.definition[field] for field in PROOF_FIELDS})


def validate_catalog(task: dict[str, Any], catalog: Catalog, registry: dict[str, Any]) -> None:
    """Fail early if a transformation violates the frozen design."""
    expected_counts = {"C0": 10, "C1": 11, "C2": 13, "C3": 11, "C4": 13, "C5": 13}
    if len(catalog.options) != expected_counts[catalog.condition]:
        raise ValueError(f"{task['task_id']} {catalog.condition} has unexpected option count")
    if len(catalog.option_ids) != len(set(catalog.option_ids)):
        raise ValueError(f"{task['task_id']} {catalog.condition} contains duplicate visible IDs")
    if not all(OPTION_ID_RE.fullmatch(option_id) for option_id in catalog.option_ids):
        raise ValueError(f"{task['task_id']} {catalog.condition} has non-neutral or unequal visible IDs")
    if len({len(option_id) for option_id in catalog.option_ids}) != 1:
        raise ValueError(f"{task['task_id']} {catalog.condition} visible IDs are not equal length")
    target_characters = registry["domains"][task["domain"]]["prompt_description_target_characters"]
    if any(len(option.display_description) != target_characters for option in catalog.options):
        raise ValueError(f"{task['task_id']} {catalog.condition} has unmatched visible description load")
    expected_targets = _counterbalance_targets(
        task,
        catalog.condition,
        registry["domains"][task["domain"]],
        registry["actions"],
        registry["alias_option_ids"],
    )
    if catalog.position_scheme != "target-position-latin-v2" or catalog.counterbalance_targets != expected_targets:
        raise ValueError(f"{task['task_id']} {catalog.condition} does not use the frozen target-position plan")
    actual_positions = {option_id: catalog.option_ids.index(option_id) for option_id in expected_targets}
    if actual_positions != expected_targets:
        raise ValueError(f"{task['task_id']} {catalog.condition} misplaced counterbalanced target option(s)")

    base_ids = registry["domains"][task["domain"]]["canonical_action_ids"]
    aliases = [option for option in catalog.options if option.is_alias]
    if catalog.condition in ALIAS_CONDITIONS:
        expected_target = task["gold_action_id"] if catalog.condition in {"C1", "C2"} else task["wrong_competitor_action_id"]
        expected_aliases = 1 if catalog.condition in {"C1", "C3"} else 3
        if len(aliases) != expected_aliases or any(option.action_id != expected_target for option in aliases):
            raise ValueError(f"{task['task_id']} {catalog.condition} has the wrong alias target")
        canonical = next(option for option in catalog.options if option.action_id == expected_target and not option.is_alias)
        if any(_proof_signature(alias) != _proof_signature(canonical) for alias in aliases):
            raise ValueError(f"{task['task_id']} {catalog.condition} aliases differ from their canonical action")
    elif aliases:
        raise ValueError(f"{task['task_id']} {catalog.condition} unexpectedly contains aliases")

    action_counts = {action_id: 0 for action_id in base_ids}
    for option in catalog.options:
        if option.action_id in action_counts:
            action_counts[option.action_id] += 1
    if any(count < 1 for count in action_counts.values()):
        raise ValueError(f"{task['task_id']} {catalog.condition} lost a canonical action")
    if catalog.condition == "C5":
        controls = registry["domains"][task["domain"]]["control_action_ids"]
        present = {option.action_id for option in catalog.options}
        if set(controls) - present:
            raise ValueError(f"{task['task_id']} C5 omitted a distinct load-control action")
        if set(controls) & set(base_ids):
            raise ValueError(f"{task['task_id']} C5 control action is not genuinely distinct")

    expected_added_load = {
        "C0": 0,
        "C1": target_characters,
        "C2": 3 * target_characters,
        "C3": target_characters,
        "C4": 3 * target_characters,
        "C5": 3 * target_characters,
    }[catalog.condition]
    if added_description_characters(catalog) != expected_added_load:
        raise ValueError(f"{task['task_id']} {catalog.condition} has unmatched added description load")


def shared_stimulus(task: dict[str, Any], catalog: Catalog) -> dict[str, Any]:
    """The model-neutral information supplied to both adapters."""
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "task_text": task["task_text"],
        "repository_context": task["repository_context"],
        "candidate_tools_in_display_order": catalog.model_options(),
    }


def stimulus_hash(task: dict[str, Any], catalog: Catalog) -> str:
    return sha256_json(shared_stimulus(task, catalog))


def added_description_characters(catalog: Catalog) -> int:
    """Report, rather than overclaim, the textual load introduced by a condition."""
    return sum(len(option.display_description) for option in catalog.options if option.option_id in {"tool_11", "tool_12", "tool_13"})
