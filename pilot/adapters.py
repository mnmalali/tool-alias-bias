"""OpenRouter and deterministic-local adapters with one normalized ledger shape."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from .common import sha256_text
from .registry import Catalog, shared_stimulus


JEV_MODEL = "typesafe/jev-1.13"
LUNA_MODEL = "openai/gpt-6-luna-decisions"
HAIKU_MODEL = "anthropic/claude-haiku-5.5"
HAIKU_PROVIDER_ONLY = "google-vertex/global"
OPENROUTER_ROOT = "https://openrouter.ai"


def returned_model_is_compatible(requested: str, returned: str | None) -> bool:
    """Accept the documented dated snapshot under a requested release ID."""
    if returned == requested:
        return True
    return bool(
        isinstance(returned, str)
        and returned.startswith(requested + "-")
        and re.fullmatch(r"\d{8}", returned[len(requested) + 1:])
    )


@dataclass
class NormalizedResponse:
    model_arm: str
    requested_model: str
    returned_model: str | None
    provider: str | None
    request_id: str | None
    selected_option: str | None
    probabilities: dict[str, Any] | None
    confidence: Any | None
    distribution_valid: bool
    validation_errors: list[str]
    usage: dict[str, Any]
    latency_ms: float
    request_payload: dict[str, Any]
    wire_request_body: str
    wire_request_hash: str
    response_status_code: int | None
    response_headers: dict[str, str]
    system_fingerprint: str | None
    raw_response: Any
    error: dict[str, Any] | None
    synthetic: bool = False
    reported_probabilities: dict[str, Any] | None = None
    reported_probability_sum: float | None = None
    probability_transform: str | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


class OpenRouterRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.retryable = retryable


@dataclass(frozen=True)
class HttpJsonResponse:
    body: dict[str, Any]
    status_code: int
    headers: dict[str, str]


RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524, 529})


def wire_json(payload: dict[str, Any]) -> str:
    """Serialize an outbound payload without sorting its experimental key order."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    try:
        return {str(key): str(item) for key, item in value.items()}
    except (AttributeError, TypeError):
        return {}


def _usage(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    value = raw.get("usage")
    return value if isinstance(value, dict) else {}


def _provider(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    direct = raw.get("provider")
    if isinstance(direct, str):
        return direct
    metadata = raw.get("openrouter_metadata")
    if isinstance(metadata, dict):
        for key in ("provider_name", "provider", "routed_provider"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
        endpoints = metadata.get("endpoints")
        endpoint_list = endpoints.get("available") if isinstance(endpoints, dict) else endpoints
        if isinstance(endpoint_list, list):
            selected = [entry for entry in endpoint_list if isinstance(entry, dict) and entry.get("selected") is True]
            for endpoint in selected:
                if not isinstance(endpoint, dict):
                    continue
                for key in ("provider_name", "provider", "routed_provider"):
                    value = endpoint.get(key)
                    if isinstance(value, str):
                        return value
    return None


def _returned_model(raw: Any) -> str | None:
    return raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), str) else None


def _system_fingerprint(raw: Any) -> str | None:
    return raw.get("system_fingerprint") if isinstance(raw, dict) and isinstance(raw.get("system_fingerprint"), str) else None


def _request_id(raw: Any, headers: dict[str, str]) -> str | None:
    if isinstance(raw, dict) and isinstance(raw.get("id"), str):
        return raw["id"]
    lowered = {key.lower(): value for key, value in headers.items()}
    for key in ("x-generation-id", "x-request-id"):
        if isinstance(lowered.get(key), str):
            return lowered[key]
    return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_decision(
    selected_option: Any, probabilities: Any, confidence: Any, expected_option_ids: list[str], *,
    confidence_required: bool, require_unit_sum: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate exact keys and bounds; native vectors also need unit mass."""
    errors: list[str] = []
    cleaned: dict[str, Any] | None = probabilities if isinstance(probabilities, dict) else None
    expected = set(expected_option_ids)
    if not isinstance(selected_option, str) or selected_option not in expected:
        errors.append("selected_option is absent or not one of the visible candidate IDs")
    if cleaned is None:
        errors.append("probabilities is not an object")
    else:
        keys = set(cleaned)
        if keys != expected:
            errors.append(f"probability keys differ from visible options (missing={sorted(expected - keys)}, extra={sorted(keys - expected)})")
        bad_values = [key for key, value in cleaned.items() if not _number(value) or not 0.0 <= float(value) <= 1.0]
        if bad_values:
            errors.append(f"probabilities outside finite [0,1] range: {sorted(bad_values)}")
        elif keys == expected and require_unit_sum:
            # JSON numbers are decimal text. Check the inclusive 0.01 rounding
            # allowance in decimal arithmetic so a displayed 0.99 is not
            # rejected by binary floating-point summation at the boundary.
            total = sum((Decimal(str(value)) for value in cleaned.values()), Decimal(0))
            if abs(total - Decimal(1)) > Decimal("0.01"):
                errors.append(f"probabilities sum to {total:.8f}, not approximately 1")
    if confidence_required and (not _number(confidence) or not 0.0 <= float(confidence) <= 1.0):
        errors.append("confidence is absent or outside finite [0,1]")
    return cleaned, errors


def normalize_haiku_weights(
    selected_option: Any, reported: Any, confidence: Any, expected_option_ids: list[str],
) -> tuple[dict[str, float] | None, float | None, list[str]]:
    """Turn a complete elicited relative-weight vector into a unit vector."""
    weights, errors = validate_decision(
        selected_option, reported, confidence, expected_option_ids,
        confidence_required=True, require_unit_sum=False,
    )
    if errors or weights is None:
        return None, None, errors
    total = sum((Decimal(str(value)) for value in weights.values()), Decimal(0))
    if total <= 0:
        return None, float(total), ["Haiku relative weights have zero total mass"]
    normalized = {option_id: float(Decimal(str(value)) / total) for option_id, value in weights.items()}
    return normalized, float(total), []


def _post_json(api_key: str, path: str, payload: dict[str, Any], timeout_seconds: float) -> HttpJsonResponse:
    # Dict insertion order is experimental state here: it carries the frozen,
    # counterbalanced candidate order for native decision criteria. Hash helpers
    # may sort keys, but the actual wire payload must never do so.
    encoded = wire_json(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OPENROUTER_ROOT}{path}",
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status_code = int(getattr(response, "status", response.getcode()))
            response_headers = _headers(response.headers)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        raise OpenRouterRequestError(
            f"OpenRouter HTTP {error.code}",
            status_code=error.code,
            body=parsed,
            headers=_headers(error.headers),
            retryable=error.code in RETRYABLE_HTTP_STATUSES,
        ) from error
    except urllib.error.URLError as error:
        raise OpenRouterRequestError(f"OpenRouter transport error: {error.reason}", retryable=True) from error
    except (TimeoutError, socket.timeout) as error:
        raise OpenRouterRequestError("OpenRouter request timed out", retryable=True) from error
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise OpenRouterRequestError(
            "OpenRouter returned non-JSON response", status_code=status_code, body=body, headers=response_headers
        ) from error
    if not isinstance(parsed, dict):
        raise OpenRouterRequestError(
            "OpenRouter returned a JSON value other than an object",
            status_code=status_code,
            body=parsed,
            headers=response_headers,
        )
    return HttpJsonResponse(body=parsed, status_code=status_code, headers=response_headers)


class BaseAdapter:
    model_arm: str
    requested_model: str

    def decide(self, task: dict[str, Any], catalog: Catalog) -> NormalizedResponse:
        raise NotImplementedError


class OpenRouterJevAdapter(BaseAdapter):
    model_arm = "jev"
    requested_model = JEV_MODEL

    def __init__(self, api_key: str, timeout_seconds: float = 60.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _payload(self, task: dict[str, Any], catalog: Catalog) -> dict[str, Any]:
        stimulus = shared_stimulus(task, catalog)
        criteria = {entry["option_id"]: entry["description"] for entry in stimulus["candidate_tools_in_display_order"]}
        return {
            "model": self.requested_model,
            "state": {
                "task": stimulus["task_text"],
                "repository_context": stimulus["repository_context"],
            },
            "questions": {
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Choose the single best next action for the task. Candidate IDs are neutral labels, not semantic hints. "
                        "Use the candidate descriptions and the stated task; select clarification when essential target information is absent."
                    ),
                    "criteria": criteria,
                }
            },
        }

    def decide(self, task: dict[str, Any], catalog: Catalog) -> NormalizedResponse:
        payload = self._payload(task, catalog)
        started = time.perf_counter()
        raw: Any = None
        http_response: HttpJsonResponse | None = None
        try:
            http_response = _post_json(self.api_key, "/api/alpha/decisions", payload, self.timeout_seconds)
            raw = http_response.body
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            answer = raw.get("answers", {}).get("next_action") if isinstance(raw.get("answers"), dict) else None
            answer = answer if isinstance(answer, dict) else {}
            probabilities, validation_errors = validate_decision(
                answer.get("choice"), answer.get("probabilities"), answer.get("confidence"), catalog.option_ids, confidence_required=True
            )
            return NormalizedResponse(
                model_arm=self.model_arm,
                requested_model=self.requested_model,
                returned_model=_returned_model(raw),
                provider=_provider(raw),
                request_id=_request_id(raw, http_response.headers),
                selected_option=answer.get("choice") if isinstance(answer.get("choice"), str) else None,
                probabilities=probabilities,
                confidence=answer.get("confidence"),
                distribution_valid=not validation_errors,
                validation_errors=validation_errors,
                usage=_usage(raw),
                latency_ms=latency_ms,
                request_payload=payload,
                wire_request_body=wire_json(payload),
                wire_request_hash=sha256_text(wire_json(payload)),
                response_status_code=http_response.status_code,
                response_headers=http_response.headers,
                system_fingerprint=_system_fingerprint(raw),
                raw_response=raw,
                error=None,
            )
        except OpenRouterRequestError as error:
            return _error_response(self.model_arm, self.requested_model, payload, started, error)
        except Exception as error:
            return _error_response(
                self.model_arm,
                self.requested_model,
                payload,
                started,
                error,
                raw_response=raw,
                http_response=http_response,
            )


class OpenRouterLunaAdapter(OpenRouterJevAdapter):
    """The same OpenRouter typed-choice contract with a pinned Luna model ID."""

    model_arm = "luna"
    requested_model = LUNA_MODEL


def _haiku_schema(option_ids: list[str]) -> dict[str, Any]:
    """Use a stable, Claude-compatible schema for the catalog's exact IDs."""
    # Display order belongs in the prompt. Canonical schema order avoids a new
    # grammar compilation for every shuffled catalog of the same size.
    schema_ids = sorted(option_ids)
    probability_properties = {
        option_id: {"type": "number", "description": "Nonnegative relative weight from 0 to 1."}
        for option_id in schema_ids
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_option": {
                "type": "string",
                "enum": schema_ids,
                "description": "Exactly one supplied candidate ID.",
            },
            "probabilities": {
                "type": "object",
                "description": "One relative weight from 0 to 1 for every supplied candidate ID; exact sum is not required.",
                "properties": probability_properties,
                "required": schema_ids,
                "additionalProperties": False,
            },
            "confidence": {"type": "number", "description": "Your confidence in the selected option, from 0 to 1."},
        },
        "required": ["selected_option", "probabilities", "confidence"],
    }


def _render_haiku_prompt(task: dict[str, Any], catalog: Catalog) -> str:
    stimulus = shared_stimulus(task, catalog)
    option_lines = "\n".join(
        f"- {entry['option_id']}: {entry['description']}" for entry in stimulus["candidate_tools_in_display_order"]
    )
    return (
        "Select the single best next action from the candidate tools.\n\n"
        f"Repository context:\n{stimulus['repository_context']}\n\n"
        f"Task:\n{stimulus['task_text']}\n\n"
        f"Candidate tools in display order:\n{option_lines}\n\n"
        "Candidate IDs are neutral labels. Do not infer their meaning from the ID. Return every candidate ID exactly once in "
        "probabilities as nonnegative relative weights from 0 to 1. They do not need to sum exactly to 1; "
        "the evaluator will divide each by their positive total. Select clarification when essential target information is absent."
    )


def _content_to_json(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise ValueError("Chat completion contains no JSON text content")
    return json.loads(content)


class OpenRouterHaikuAdapter(BaseAdapter):
    model_arm = "haiku"
    requested_model = HAIKU_MODEL

    def __init__(self, api_key: str, timeout_seconds: float = 60.0, temperature: float = 0.0) -> None:
        if temperature != 0.0:
            raise ValueError("Haiku 5.5's pinned OpenRouter endpoint does not support temperature; use the default")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _payload(self, task: dict[str, Any], catalog: Catalog) -> dict[str, Any]:
        return {
            "model": self.requested_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You make one bounded tool-selection decision. Follow the supplied JSON schema exactly. "
                        "Do not use option labels as semantic evidence."
                    ),
                },
                {"role": "user", "content": _render_haiku_prompt(task, catalog)},
            ],
            "max_tokens": 300,
            "reasoning": {"enabled": False},
            "provider": {"only": [HAIKU_PROVIDER_ONLY], "require_parameters": True, "allow_fallbacks": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "tool_selection", "strict": True, "schema": _haiku_schema(catalog.option_ids)},
            },
        }

    def decide(self, task: dict[str, Any], catalog: Catalog) -> NormalizedResponse:
        payload = self._payload(task, catalog)
        started = time.perf_counter()
        raw: dict[str, Any] | None = None
        http_response: HttpJsonResponse | None = None
        try:
            http_response = _post_json(self.api_key, "/api/v1/chat/completions", payload, self.timeout_seconds)
        except OpenRouterRequestError as error:
            return _error_response(self.model_arm, self.requested_model, payload, started, error)
        except Exception as error:
            # Preserve a normalized ledger row even if a future transport
            # implementation raises an unexpected exception before returning
            # its response wrapper.  It may still represent an issued request.
            return _error_response(self.model_arm, self.requested_model, payload, started, error)
        raw = http_response.body
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            choices = raw.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
            if not isinstance(message, dict):
                raise ValueError("Chat completion contains no assistant message")
            answer = _content_to_json(message.get("content"))
            if not isinstance(answer, dict):
                raise ValueError("Structured output is not a JSON object")
            probabilities, reported_sum, validation_errors = normalize_haiku_weights(
                answer.get("selected_option"), answer.get("probabilities"), answer.get("confidence"), catalog.option_ids,
            )
            return NormalizedResponse(
                model_arm=self.model_arm,
                requested_model=self.requested_model,
                returned_model=_returned_model(raw),
                provider=_provider(raw),
                request_id=_request_id(raw, http_response.headers),
                selected_option=answer.get("selected_option") if isinstance(answer.get("selected_option"), str) else None,
                probabilities=probabilities,
                confidence=answer.get("confidence"),
                distribution_valid=not validation_errors,
                validation_errors=validation_errors,
                usage=_usage(raw),
                latency_ms=latency_ms,
                request_payload=payload,
                wire_request_body=wire_json(payload),
                wire_request_hash=sha256_text(wire_json(payload)),
                response_status_code=http_response.status_code,
                response_headers=http_response.headers,
                system_fingerprint=_system_fingerprint(raw),
                raw_response=raw,
                error=None,
                reported_probabilities=answer.get("probabilities") if isinstance(answer.get("probabilities"), dict) else None,
                reported_probability_sum=reported_sum,
                probability_transform="haiku_relative_weights_l1" if not validation_errors else None,
            )
        except (ValueError, json.JSONDecodeError, TypeError, IndexError) as error:
            return _parse_error_response(
                self.model_arm, self.requested_model, payload, started, raw, http_response, error
            )
        except Exception as error:
            return _error_response(
                self.model_arm,
                self.requested_model,
                payload,
                started,
                error,
                raw_response=raw,
                http_response=http_response,
            )


def _parse_error_response(
    model_arm: str,
    requested_model: str,
    payload: dict[str, Any],
    started: float,
    raw: dict[str, Any],
    http_response: HttpJsonResponse,
    error: Exception,
) -> NormalizedResponse:
    """Retain a billed HTTP-success response whose structured content is unusable."""
    wire_body = wire_json(payload)
    return NormalizedResponse(
        model_arm=model_arm,
        requested_model=requested_model,
        returned_model=_returned_model(raw),
        provider=_provider(raw),
        request_id=_request_id(raw, http_response.headers),
        selected_option=None,
        probabilities=None,
        confidence=None,
        distribution_valid=False,
        validation_errors=[str(error)],
        usage=_usage(raw),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        request_payload=payload,
        wire_request_body=wire_body,
        wire_request_hash=sha256_text(wire_body),
        response_status_code=http_response.status_code,
        response_headers=http_response.headers,
        system_fingerprint=_system_fingerprint(raw),
        raw_response=raw,
        error={
            "type": "ResponseParseError",
            "message": str(error),
            "status_code": http_response.status_code,
            "retryable": False,
        },
    )


def _error_response(
    model_arm: str,
    requested_model: str,
    payload: dict[str, Any],
    started: float,
    error: Exception,
    *,
    raw_response: Any = None,
    http_response: HttpJsonResponse | None = None,
) -> NormalizedResponse:
    status_code = (
        error.status_code if isinstance(error, OpenRouterRequestError)
        else http_response.status_code if http_response is not None else None
    )
    body = error.body if isinstance(error, OpenRouterRequestError) else raw_response
    headers = error.headers if isinstance(error, OpenRouterRequestError) else http_response.headers if http_response else {}
    wire_body = wire_json(payload)
    return NormalizedResponse(
        model_arm=model_arm,
        requested_model=requested_model,
        returned_model=_returned_model(body),
        provider=_provider(body),
        request_id=_request_id(body, headers),
        selected_option=None,
        probabilities=None,
        confidence=None,
        distribution_valid=False,
        validation_errors=[str(error)],
        usage=_usage(body),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        request_payload=payload,
        wire_request_body=wire_body,
        wire_request_hash=sha256_text(wire_body),
        response_status_code=status_code,
        response_headers=headers,
        system_fingerprint=_system_fingerprint(body),
        raw_response=body,
        error={
            "type": type(error).__name__,
            "message": str(error),
            "status_code": status_code,
            "retryable": error.retryable if isinstance(error, OpenRouterRequestError) else False,
        },
    )


class MockAdapter(BaseAdapter):
    """Deterministic plumbing test only; it never represents model behavior."""

    def __init__(self, model_arm: str) -> None:
        if model_arm not in {"jev", "luna", "haiku"}:
            raise ValueError(f"Unsupported mock model arm {model_arm}")
        self.model_arm = model_arm
        self.requested_model = f"local/mock-{model_arm}"

    def decide(self, task: dict[str, Any], catalog: Catalog) -> NormalizedResponse:
        payload = {"shared_stimulus": shared_stimulus(task, catalog), "adapter": "deterministic_mock"}
        wire_body = wire_json(payload)
        # C3/C4 occasionally route to the frozen wrong competitor so the analysis path observes changes.
        selector = int(sha256_text(f"{self.model_arm}|{task['task_id']}|{catalog.condition}")[:8], 16)
        target_action = task["gold_action_id"]
        if catalog.condition in {"C3", "C4"} and selector % 3 == 0:
            target_action = task["wrong_competitor_action_id"]
        target_options = sorted(option.option_id for option in catalog.options if option.action_id == target_action)
        target_mass = 0.72 if len(target_options) >= 3 else 0.78
        remaining = [option.option_id for option in catalog.options if option.action_id != target_action]
        probabilities: dict[str, float] = {}
        for option_id in target_options:
            probabilities[option_id] = target_mass / len(target_options)
        for option_id in remaining:
            probabilities[option_id] = (1.0 - target_mass) / len(remaining)
        # Correct floating point drift in one non-target entry without altering the intended mass pattern.
        correction_key = remaining[-1] if remaining else target_options[-1]
        probabilities[correction_key] += 1.0 - sum(probabilities.values())
        selected = target_options[selector % len(target_options)]
        raw = {
            "id": f"mock-{self.model_arm}-{task['task_id']}-{catalog.condition}",
            "model": self.requested_model,
            "provider": "local-deterministic",
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0},
            "answer": {"selected_option": selected, "probabilities": probabilities, "confidence": max(probabilities.values())},
        }
        cleaned, errors = validate_decision(selected, probabilities, raw["answer"]["confidence"], catalog.option_ids, confidence_required=True)
        return NormalizedResponse(
            model_arm=self.model_arm,
            requested_model=self.requested_model,
            returned_model=self.requested_model,
            provider="local-deterministic",
            request_id=raw["id"],
            selected_option=selected,
            probabilities=cleaned,
            confidence=raw["answer"]["confidence"],
            distribution_valid=not errors,
            validation_errors=errors,
            usage=raw["usage"],
            latency_ms=0.0,
            request_payload=payload,
            wire_request_body=wire_body,
            wire_request_hash=sha256_text(wire_body),
            response_status_code=200,
            response_headers={},
            system_fingerprint="local-deterministic-v1",
            raw_response=raw,
            error=None,
            synthetic=True,
        )


def make_adapters(kind: str, model_arms: list[str], timeout_seconds: float, temperature: float) -> list[BaseAdapter]:
    if kind == "mock":
        return [MockAdapter(arm) for arm in model_arms]
    if kind != "openrouter":
        raise ValueError(f"Unknown adapter kind: {kind}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set; no live request was sent")
    adapters: list[BaseAdapter] = []
    for arm in model_arms:
        if arm == "jev":
            adapters.append(OpenRouterJevAdapter(api_key, timeout_seconds))
        elif arm == "luna":
            adapters.append(OpenRouterLunaAdapter(api_key, timeout_seconds))
        elif arm == "haiku":
            adapters.append(OpenRouterHaikuAdapter(api_key, timeout_seconds, temperature))
        else:
            raise ValueError(f"Unknown model arm: {arm}")
    return adapters
