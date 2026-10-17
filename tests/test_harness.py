import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from pilot.adapters import (
    HttpJsonResponse,
    MockAdapter,
    OpenRouterHaikuAdapter,
    OpenRouterJevAdapter,
    OpenRouterLunaAdapter,
    _parse_error_response,
    normalize_haiku_weights,
    validate_decision,
    wire_json,
)
from pilot.analysis import analyze_run
from pilot.common import append_jsonl, write_json
from pilot.registry import CONDITIONS, build_catalog, load_dataset, load_registry
from pilot.runner import _derived_policies, _retryable


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.registry = load_registry()

    def test_pilot_has_exact_required_shape(self) -> None:
        tasks = self.dataset["tasks"]
        self.assertEqual(len(tasks), 30)
        self.assertEqual(sum(task["domain"] == "repository_inspection" for task in tasks), 10)
        self.assertEqual(sum(task["domain"] == "test_selection" for task in tasks), 10)
        self.assertEqual(sum(task["domain"] == "version_history" for task in tasks), 10)

    def test_condition_counts_and_repeat_stability(self) -> None:
        task = self.dataset["tasks"][0]
        expected = {"C0": 10, "C1": 11, "C2": 13, "C3": 11, "C4": 13, "C5": 13}
        for condition in CONDITIONS:
            first = build_catalog(task, condition, self.registry)
            second = build_catalog(task, condition, self.registry)
            self.assertEqual(len(first.options), expected[condition])
            self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_catalogs_balance_targets_and_added_description_load(self) -> None:
        actions = self.registry["actions"]
        for domain, definition in self.registry["domains"].items():
            domain_tasks = [task for task in self.dataset["tasks"] if task["domain"] == domain]
            for condition in CONDITIONS:
                catalogs = [build_catalog(task, condition, self.registry) for task in domain_tasks]
                for catalog in catalogs:
                    self.assertTrue(
                        all(
                            len(option.display_description) == definition["prompt_description_target_characters"]
                            for option in catalog.options
                        )
                    )
                gold_positions = sorted(
                    catalog.option_ids.index(actions[task["gold_action_id"]]["base_option_id"])
                    for task, catalog in zip(domain_tasks, catalogs, strict=True)
                )
                wrong_positions = sorted(
                    catalog.option_ids.index(actions[task["wrong_competitor_action_id"]]["base_option_id"])
                    for task, catalog in zip(domain_tasks, catalogs, strict=True)
                )
                self.assertEqual(gold_positions, list(range(10)))
                self.assertEqual(wrong_positions, list(range(10)))

    def test_malformed_jev_response_cannot_supply_p1(self) -> None:
        task = self.dataset["tasks"][0]
        catalog = build_catalog(task, "C0", self.registry)
        malformed = SimpleNamespace(
            model_arm="jev",
            selected_option=None,
            confidence=0.9,
            distribution_valid=False,
            probabilities=None,
        )
        policies = _derived_policies(malformed, catalog)
        self.assertFalse(policies["P1"]["available"])
        self.assertIsNone(policies["P1"]["action_id"])

    def test_mock_response_derives_common_policies(self) -> None:
        task = self.dataset["tasks"][0]
        catalog = build_catalog(task, "C2", self.registry)
        response = MockAdapter("jev").decide(task, catalog)
        policies = _derived_policies(response, catalog)
        self.assertTrue(response.distribution_valid)
        self.assertTrue(policies["P1"]["available"])
        self.assertTrue(policies["P2"]["available"])
        self.assertTrue(policies["P3"]["available"])
        self.assertEqual(sum(response.probabilities.values()), 1.0)

    def test_luna_uses_native_decisions_choice_and_confidence(self) -> None:
        task = self.dataset["tasks"][0]
        catalog = build_catalog(task, "C2", self.registry)
        adapter = OpenRouterLunaAdapter("unused")
        payload = adapter._payload(task, catalog)
        self.assertEqual(payload["model"], "openai/gpt-6-luna-decisions")
        self.assertEqual(list(payload["questions"]["next_action"]["criteria"]), catalog.option_ids)
        chosen = catalog.option_ids[0]
        probabilities = {option_id: (1.0 if option_id == chosen else 0.0) for option_id in catalog.option_ids}
        raw = {"model": adapter.requested_model, "answers": {"next_action": {
            "type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": 0.9,
        }}, "usage": {"cost": 0.0002}}
        with patch("pilot.adapters._post_json", return_value=HttpJsonResponse(raw, 200, {})) as post:
            response = adapter.decide(task, catalog)
        self.assertEqual(post.call_args.args[1], "/api/alpha/decisions")
        self.assertTrue(response.distribution_valid)
        self.assertTrue(_derived_policies(response, catalog)["P1"]["available"])

    def test_malformed_distribution_is_not_repaired(self) -> None:
        probabilities, errors = validate_decision("tool_01", {"tool_01": 1.0}, 1.0, ["tool_01", "tool_02"], confidence_required=True)
        self.assertEqual(probabilities, {"tool_01": 1.0})
        self.assertTrue(errors)

        _, out_of_range = validate_decision(
            "tool_01", {"tool_01": 1.2, "tool_02": -0.2}, 1.1,
            ["tool_01", "tool_02"], confidence_required=True,
        )
        self.assertTrue(any("outside finite [0,1]" in error for error in out_of_range))
        self.assertTrue(any("confidence" in error for error in out_of_range))

    def test_distribution_sum_rounding_allowance_is_inclusive_without_repair(self) -> None:
        ids = [f"tool_{index:02d}" for index in range(1, 14)]
        rounded = {option_id: 0.01 for option_id in ids}
        rounded["tool_01"] = 0.2
        rounded["tool_08"] = 0.03
        rounded["tool_09"] = 0.5
        for option_id in ids[9:]:
            rounded[option_id] = 0.05
        # This is the complete 13-option vector returned by the live smoke.
        self.assertAlmostEqual(sum(rounded.values()), 0.99)
        validated, errors = validate_decision("tool_09", rounded, 0.6, ids, confidence_required=True)
        self.assertEqual(errors, [])
        self.assertEqual(validated, rounded)

        outside = dict(rounded)
        outside["tool_02"] = 0.009
        _, errors = validate_decision("tool_09", outside, 0.6, ids, confidence_required=True)
        self.assertTrue(any("probabilities sum" in error for error in errors))

    def test_haiku_relative_weights_are_explicitly_normalized_and_raw_values_retained(self) -> None:
        task = self.dataset["tasks"][0]
        catalog = build_catalog(task, "C2", self.registry)
        selected = catalog.option_ids[0]
        reported = {option_id: (0.5 if option_id == selected else 0.04)
                    for option_id in catalog.option_ids}
        self.assertEqual(len(reported), 13)
        native, native_errors = validate_decision(selected, reported, 0.6,
                                                  catalog.option_ids, confidence_required=True)
        self.assertEqual(native, reported)
        self.assertTrue(any("probabilities sum" in error for error in native_errors))

        normalized, reported_sum, errors = normalize_haiku_weights(
            selected, reported, 0.6, catalog.option_ids,
        )
        self.assertEqual(errors, [])
        self.assertAlmostEqual(reported_sum, 0.98)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertAlmostEqual(normalized[selected], 0.5 / 0.98)
        self.assertEqual(reported[selected], 0.5)

        raw = {"model": "anthropic/claude-haiku-5.5", "provider": "Google",
               "choices": [{"message": {"content": json.dumps({
                   "selected_option": selected, "probabilities": reported, "confidence": 0.6,
               })}}], "usage": {"cost": 0.000274}}
        with patch("pilot.adapters._post_json", return_value=HttpJsonResponse(raw, 200, {})):
            response = OpenRouterHaikuAdapter("not-a-real-key").decide(task, catalog)
        self.assertTrue(response.distribution_valid)
        self.assertEqual(response.reported_probabilities, reported)
        self.assertAlmostEqual(response.reported_probability_sum, 0.98)
        self.assertEqual(response.probability_transform, "haiku_relative_weights_l1")
        self.assertAlmostEqual(sum(response.probabilities.values()), 1.0)
        self.assertEqual(json.loads(response.raw_response["choices"][0]["message"]["content"])["probabilities"], reported)

        zero = {option_id: 0.0 for option_id in catalog.option_ids}
        _, _, zero_errors = normalize_haiku_weights(selected, zero, 0.6, catalog.option_ids)
        self.assertTrue(any("zero total mass" in error for error in zero_errors))
        missing = dict(reported)
        missing.pop(selected)
        _, _, missing_errors = normalize_haiku_weights(selected, missing, 0.6, catalog.option_ids)
        self.assertTrue(any("keys differ" in error for error in missing_errors))

    def test_catalog_order_is_shared_on_jev_wire_and_haiku_prompt(self) -> None:
        task = next(task for task in self.dataset["tasks"] if task["task_id"] == "RI-01")
        catalog = build_catalog(task, "C2", self.registry)
        expected_ids = catalog.option_ids

        jev_payload = OpenRouterJevAdapter("not-a-real-key")._payload(task, catalog)
        criteria = jev_payload["questions"]["next_action"]["criteria"]
        self.assertEqual(list(criteria), expected_ids)
        serialized_jev = wire_json(jev_payload)
        criteria_positions = [serialized_jev.index(f'"{option_id}":') for option_id in expected_ids]
        self.assertEqual(criteria_positions, sorted(criteria_positions))

        haiku_payload = OpenRouterHaikuAdapter("not-a-real-key")._payload(task, catalog)
        self.assertEqual(haiku_payload["model"], "anthropic/claude-haiku-5.5")
        self.assertNotIn("temperature", haiku_payload)
        self.assertEqual(haiku_payload["reasoning"], {"enabled": False})
        self.assertEqual(haiku_payload["provider"]["only"], ["google-vertex/global"])
        self.assertFalse(haiku_payload["provider"]["allow_fallbacks"])
        prompt = haiku_payload["messages"][1]["content"]
        rendered_ids = [line.split(":", 1)[0][2:] for line in prompt.splitlines() if line.startswith("- tool_")]
        self.assertEqual(rendered_ids, expected_ids)
        schema = haiku_payload["response_format"]["json_schema"]["schema"]
        canonical_ids = sorted(expected_ids)
        self.assertEqual(schema["properties"]["selected_option"]["enum"], canonical_ids)
        probabilities = schema["properties"]["probabilities"]
        self.assertEqual(list(probabilities["properties"]), canonical_ids)
        self.assertEqual(probabilities["required"], canonical_ids)
        self.assertFalse(probabilities["additionalProperties"])
        self.assertTrue(all("minimum" not in value and "maximum" not in value
                            for value in probabilities["properties"].values()))
        self.assertNotIn("minimum", schema["properties"]["confidence"])
        self.assertEqual(schema, OpenRouterHaikuAdapter("not-a-real-key")._payload(
            task, build_catalog(task, "C4", self.registry)
        )["response_format"]["json_schema"]["schema"])

    def test_haiku_rejects_unsupported_nondefault_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support temperature"):
            OpenRouterHaikuAdapter("not-a-real-key", temperature=0.5)

    def test_parse_failure_preserves_a_potentially_billed_http_response(self) -> None:
        payload = {"model": "example"}
        raw = {
            "id": "generation-1",
            "model": "anthropic/claude-haiku-5.5",
            "provider": "example-provider",
            "usage": {"cost": 0.0042},
        }
        response = _parse_error_response(
            "haiku",
            "anthropic/claude-haiku-5.5",
            payload,
            0.0,
            raw,
            HttpJsonResponse(body=raw, status_code=200, headers={"X-Generation-Id": "generation-1"}),
            ValueError("bad JSON"),
        )
        self.assertEqual(response.raw_response, raw)
        self.assertEqual(response.usage, {"cost": 0.0042})
        self.assertEqual(response.request_id, "generation-1")
        self.assertFalse(response.error["retryable"])

    def test_unexpected_haiku_transport_exception_becomes_a_ledgerable_response(self) -> None:
        task = self.dataset["tasks"][0]
        catalog = build_catalog(task, "C0", self.registry)
        with patch("pilot.adapters._post_json", side_effect=RuntimeError("unexpected transport failure")):
            response = OpenRouterHaikuAdapter("not-a-real-key").decide(task, catalog)
        self.assertEqual(response.error["type"], "RuntimeError")
        self.assertFalse(response.error["retryable"])
        self.assertIsNone(response.raw_response)

    def test_documented_decisions_and_chat_shapes_normalize_without_network(self) -> None:
        task = self.dataset["tasks"][0]
        for condition in ("C0", "C2"):
            catalog = build_catalog(task, condition, self.registry)
            selected = catalog.option_ids[0]
            probabilities = {option_id: (0.8 if option_id == selected else 0.2 / (len(catalog.option_ids) - 1))
                             for option_id in catalog.option_ids}
            jev_raw = {
                "id": "gen-dec-offline-example", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
                "answers": {"next_action": {"type": "choice", "choice": selected, "confidence": 0.8,
                                            "probabilities": probabilities}},
                "usage": {"input_tokens": 476, "output_tokens": 0, "cost": 0.000019992},
            }
            with patch("pilot.adapters._post_json", return_value=HttpJsonResponse(body=jev_raw, status_code=200, headers={})):
                jev = OpenRouterJevAdapter("not-a-real-key").decide(task, catalog)
            self.assertTrue(jev.distribution_valid)
            self.assertEqual(jev.selected_option, selected)
            self.assertEqual(jev.returned_model, "typesafe/jev-1.13-20260917")
            self.assertEqual(jev.usage["cost"], 0.000019992)

            chat_raw = {
                "id": "gen-chat-offline-example", "model": "anthropic/claude-haiku-5.5",
                "openrouter_metadata": {"endpoints": {"available": [
                    {"provider": "Anthropic", "selected": True},
                ]}},
                "choices": [{"message": {"content": json.dumps({
                    "selected_option": selected, "probabilities": probabilities, "confidence": 0.8,
                })}}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 110, "cost": 0.000125},
            }
            with patch("pilot.adapters._post_json", return_value=HttpJsonResponse(body=chat_raw, status_code=200, headers={})):
                haiku = OpenRouterHaikuAdapter("not-a-real-key").decide(task, catalog)
            self.assertTrue(haiku.distribution_valid)
            self.assertEqual(haiku.selected_option, selected)
            self.assertEqual(haiku.provider, "Anthropic")
            self.assertEqual(haiku.usage["prompt_tokens"], 700)

    def test_analysis_reports_physical_retries_errors_and_incomplete_requests(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_json(run_dir / "manifest.json", {"planned_requests": 3})
            rows = [
                {
                    "logical_request_id": "one",
                    "transport_attempt": 0,
                    "is_final_attempt": False,
                    "model_arm": "jev",
                    "task_id": "RI-01",
                    "condition": "C0",
                    "repeat": 1,
                    "error": {"type": "Timeout"},
                    "usage": {},
                },
                {
                    "logical_request_id": "one",
                    "transport_attempt": 1,
                    "is_final_attempt": True,
                    "model_arm": "jev",
                    "task_id": "RI-01",
                    "condition": "C0",
                    "repeat": 1,
                    "error": None,
                    "usage": {"cost": 0},
                },
                {
                    "logical_request_id": "two",
                    "transport_attempt": 0,
                    "is_final_attempt": True,
                    "model_arm": "jev",
                    "task_id": "RI-02",
                    "condition": "C0",
                    "repeat": 1,
                    "error": {"type": "ParseError"},
                    "usage": {},
                },
            ]
            for row in rows:
                append_jsonl(run_dir / "ledger.jsonl", row)
            accounting = analyze_run(run_dir)["request_accounting"]
            self.assertEqual(accounting["physical_error_attempts"], 2)
            self.assertEqual(accounting["retry_attempts"], 1)
            self.assertEqual(accounting["incomplete_logical_requests"], 1)

    def test_only_explicitly_transient_errors_are_retryable(self) -> None:
        self.assertTrue(_retryable(SimpleNamespace(error={"retryable": True})))
        self.assertFalse(_retryable(SimpleNamespace(error={"retryable": False})))
        self.assertFalse(_retryable(SimpleNamespace(error={"type": "ResponseParseError"})))


if __name__ == "__main__":
    unittest.main()
