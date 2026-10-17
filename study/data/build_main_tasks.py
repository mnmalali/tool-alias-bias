"""Materialize the prespecified synthetic main-study task set.

The scenario seeds below were authored before any paid model calls. Three
questions per seed share a scenario-family ID; the family, not the question or
API repeat, is the independent resampling unit. The output is checked into the
project so a run does not silently regenerate different text.
"""

from __future__ import annotations

import json
import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "pilot" / "data" / "pilot_tasks.json"
OUTPUT = ROOT / "study" / "data" / "main_tasks.json"
REVIEW = ROOT / "study" / "data" / "main_label_review.csv"

# Each action has one calibration family and two held-out families. Each family
# is expressed as (slug, incident context, concrete information need).
SCENARIOS = {
    "repository_inspection": {
        "repo.read_service": [
            ("shortage-guard", "A reservation for more units than are available is rejected", "inspect the code path that raises the shortage error"),
            ("missing-sku-stock", "A checkout requests a SKU absent from the in-memory stock map", "inspect how missing stock is treated before a reservation"),
            ("release-validation", "A reconciliation job tries to release zero held units", "inspect the implementation that validates a release quantity"),
        ],
        "repo.read_pricing": [
            ("fractional-tax", "A taxable order produces a total with a fractional cent", "inspect the rounding mode applied to the customer total"),
            ("decimal-conversion", "A finance reviewer questions how a floating tax rate enters the money calculation", "inspect the decimal conversion used in tax calculation"),
            ("whole-cent-total", "A receipt must show an integer number of cents after tax", "inspect the function that returns the final whole-cent total"),
        ],
        "repo.read_api": [
            ("queued-order", "An API caller submits an order and expects a queued response", "inspect the helper that constructs the order-intake payload"),
            ("health-status", "A readiness probe reports the service status", "inspect the implementation of the health response"),
            ("health-service-name", "Monitoring checks which service name appears in a health payload", "inspect the API helper that sets the service identity"),
        ],
        "repo.read_config": [
            ("alert-default", "An operator has not set the low-stock alert environment variable", "inspect the default alert threshold and its parser"),
            ("negative-alert", "A deployment supplies a negative low-stock alert threshold", "inspect the validation of the configured limit"),
            ("alert-override", "Two environments use different STOCK_ALERT_LIMIT values", "inspect how the runtime setting is read and converted"),
        ],
        "repo.read_webhooks": [
            ("transient-delivery", "A restock notification fails on its first network attempt and later succeeds", "inspect the retry loop for notification delivery"),
            ("exhausted-delivery", "All allowed restock-notification attempts fail", "inspect the error raised after retry exhaustion"),
            ("duplicate-restock", "An event source redelivers the same restock event identifier", "inspect the implementation that prevents a second stock increment"),
        ],
        "repo.read_cli": [
            ("compact-json", "An automation consumer expects a compact inventory JSON string", "inspect the inventory output formatter and its JSON separators"),
            ("plain-text-stock", "A terminal user sees a stock record as text rather than JSON", "inspect the branch that renders the plain-text inventory record"),
            ("unsupported-format", "A caller requests an output format other than text or JSON", "inspect where the formatter rejects unsupported formats"),
        ],
        "repo.read_readme": [
            ("test-command-doc", "A new contributor needs the documented command for the fixture test suite", "inspect the contributor document that publishes the command"),
            ("fixture-purpose", "A reviewer wants the repository's own statement of what CopperLedger is", "inspect the top-level document describing the fixture's purpose"),
            ("dependency-note", "A maintainer asks whether the fixture declares third-party runtime dependencies", "inspect the repository overview that states its dependency policy"),
        ],
        "repo.read_service_tests": [
            ("shortage-expectation", "A change to reservation errors needs its existing focused test contract", "inspect the unit tests asserting shortage behavior"),
            ("available-stock-test", "A developer wants to see the focused expectation for an unknown SKU's available stock", "inspect the service test file that asserts the default quantity"),
            ("release-test", "A maintainer wants the existing focused test for returning reserved units", "inspect the service test file containing the release expectation"),
        ],
        "repo.search_reserve_stock": [
            ("reservation-callers", "A refactor needs both the reservation definition and source call sites", "search source for the reserve_stock symbol across files"),
            ("checkout-dependency", "An engineer needs to locate where checkout invokes reservation logic", "find the reserve_stock occurrence in the checkout flow"),
            ("cross-file-usage", "A reviewer wants a source-wide listing of reserve_stock references", "locate the symbol in both its implementation and caller"),
        ],
        "repo.request_clarification": [
            ("vague-inspection", "A ticket says only 'inspect the repository' without naming a target or question", "obtain the missing inspection target and concrete question"),
            ("empty-review", "A request says 'look into this' but supplies no file, symbol, or behavior", "ask what part of the repository should be examined"),
            ("unspecified-code", "A teammate asks for a code review without identifying any code or concern", "clarify the target and purpose before opening a file"),
        ],
    },
    "test_selection": {
        "test.service_unit": [
            ("shortage-unit", "A reservation can now overdraw a SKU with one unit left", "exercise the focused shortage-rejection unit test"),
            ("negative-balance", "A stock guard might permit a request that leaves a negative balance", "check the service unit suite's over-reservation assertion"),
            ("insufficient-error", "A caller no longer receives the insufficient-stock exception for an oversized hold", "verify the service-level shortage behavior"),
        ],
        "test.pricing_unit": [
            ("half-cent", "A customer total at an exact half-cent rounds the wrong way", "exercise the pricing unit test for half-up rounding"),
            ("bankers-rounding", "A tax change may have replaced commercial rounding with ties-to-even", "verify the focused half-cent pricing assertion"),
            ("receipt-cent", "A small taxable amount produces one cent less than expected", "check the pricing unit suite's rounding edge case"),
        ],
        "test.api_unit": [
            ("health-identity", "The health endpoint may announce the wrong service name", "exercise the API unit test for health identity"),
            ("readiness-payload", "A readiness payload still returns OK but has the wrong service field", "verify the focused health-response contract"),
            ("monitoring-parser", "Monitoring rejects the inventory health payload after a response edit", "check the API unit suite's health assertion"),
        ],
        "test.webhooks_unit": [
            ("transient-retry", "The first notification send raises a transient connection error", "exercise the webhook unit test for retrying delivery"),
            ("third-attempt", "A notifier stops before the third permitted send attempt", "verify the focused delivery-retry assertion"),
            ("retry-loop-bound", "A retry-loop change may omit its last allowed attempt", "check the webhook unit suite's transient-failure case"),
        ],
        "test.cli_unit": [
            ("json-compactness", "An automation parser rejects new spaces in the inventory JSON", "exercise the CLI unit test for compact JSON rendering"),
            ("json-separators", "A formatter change alters commas and colons in the JSON output", "verify the focused CLI JSON assertion"),
            ("machine-output", "Machine-readable inventory output gains unexpected whitespace", "check the CLI unit suite's automation format case"),
        ],
        "test.checkout_integration": [
            ("checkout-receipt", "A checkout receipt reports the subtotal in the total field", "exercise the integration test joining stock hold and taxed total"),
            ("hold-and-tax", "A purchase must reserve one unit and return the taxed receipt together", "verify the end-to-end checkout assertion"),
            ("total-cents", "The checkout flow computes tax but serializes the wrong total_cents value", "check the checkout integration suite's receipt contract"),
        ],
        "test.restock_integration": [
            ("restock-availability", "A restock event changes stock but checkout still sees the old quantity", "exercise the integration test connecting restock to checkout availability"),
            ("availability-handoff", "The availability view stays at zero after a valid restock", "verify the cross-component restock handoff"),
            ("visible-stock", "A checkout-visible stock query ignores a preceding restock event", "check the restock integration suite's availability assertion"),
        ],
        "test.webhook_dedup_integration": [
            ("duplicate-event", "At-least-once delivery sends the same restock event twice", "exercise the integration test that forbids double counting"),
            ("event-id-replay", "Replaying an already processed event ID increases inventory again", "verify the duplicate-webhook stock invariant"),
            ("idempotent-restock", "The second copy of one restock event should leave quantity unchanged", "check the webhook integration suite's deduplication case"),
        ],
        "test.reservation_regression": [
            ("zero-release", "A reconciliation path might permit releasing zero held units", "exercise the reservation regression test for zero release"),
            ("nonpositive-release", "A boundary check was loosened from nonpositive to negative only", "verify the focused release-quantity regression"),
            ("reconciliation-guard", "A patch may silently accept a zero-unit inventory reconciliation", "check the regression suite's invalid-release assertion"),
        ],
        "test.request_clarification": [
            ("unspecified-tests", "A ticket says only 'run the relevant tests' without naming a change", "ask which component and behavior need testing"),
            ("missing-change", "A developer requests a focused test but gives no changed code or symptom", "obtain the missing component and expected behavior"),
            ("vague-verification", "The handoff says 'verify it works' with no described functionality", "clarify the test target before executing a suite"),
        ],
    },
    "version_history": {
        "history.show_reserve_introduction": [
            ("reserve-intro-patch", "An audit needs the tagged commit that first introduced reservation behavior", "show the reservation-introduction patch and commit metadata"),
            ("reserve-intro-stat", "A reviewer wants the changed-file stat for the reserve-stock introduction", "show the tagged introduction commit with its stat"),
            ("reserve-intro-author", "A change log needs the author and timestamp of the reservation-introduction patch", "open the tagged reservation commit in full"),
        ],
        "history.show_price_rounding": [
            ("rounding-fix-patch", "A finance audit needs the tagged price-rounding fix", "show the price-rounding commit patch and metadata"),
            ("rounding-fix-stat", "A reviewer asks which files changed in the tagged rounding correction", "show the rounding-fix commit with its stat"),
            ("rounding-fix-author", "The team needs the author and date of the price-rounding correction", "open the full tagged rounding-fix commit"),
        ],
        "history.show_webhook_retry": [
            ("retry-commit", "An incident review needs the tagged commit that added webhook retries", "show the webhook-retry patch and commit metadata"),
            ("retry-stat", "A maintainer asks for the changed-file stat of the retry change", "show the tagged retry commit with its stat"),
            ("retry-author", "The release notes need the author and date of the webhook-retry patch", "open the full tagged retry commit"),
        ],
        "history.show_api_health": [
            ("health-commit", "An audit needs the tagged introduction of the API health helper", "show the API-health patch and commit metadata"),
            ("health-stat", "A reviewer asks which files the health-endpoint change touched", "show the tagged API-health commit stat"),
            ("health-author", "Release notes need the author and timestamp for the health response change", "open the full tagged API-health commit"),
        ],
        "history.show_cli_json": [
            ("cli-json-commit", "A formatter review needs the tagged patch that added CLI JSON output", "show the CLI-JSON commit patch and metadata"),
            ("cli-json-stat", "A maintainer asks for the changed-file stat of the JSON formatter change", "show the tagged CLI-JSON commit stat"),
            ("cli-json-author", "Release notes need the author and timestamp of the CLI JSON feature", "open the full tagged CLI-JSON commit"),
        ],
        "history.show_config_default": [
            ("config-commit", "An operator needs the tagged patch that set the alert-limit default", "show the configuration-default commit and its stat"),
            ("config-files", "A reviewer asks which files were changed by the alert-default patch", "show the tagged config-default commit stat"),
            ("config-author", "An audit needs the author and date of the configuration-default change", "open the full tagged config-default commit"),
        ],
        "history.diff_reservation_regression": [
            ("regression-diff", "A bug review needs the service.py change between the config-default and reservation-regression tags", "diff the bounded service file across those two revisions"),
            ("release-guard-diff", "A reviewer asks for the exact line that added positive-release validation", "show the bounded service.py diff for the regression fix"),
            ("before-after", "The team needs a before-and-after patch for the release quantity guard", "compare service.py at the two named fixture revisions"),
        ],
        "history.blame_service": [
            ("current-line-author", "An owner needs line-level attribution for the current reservation definition", "blame the current service.py lines"),
            ("guard-attribution", "A maintainer asks which commit last touched the current shortage guard", "inspect line-level attribution for service.py"),
            ("release-attribution", "An incident report needs the author of the current release validation lines", "blame the present service implementation"),
        ],
        "history.log_webhooks": [
            ("webhook-timeline", "A reviewer needs every commit affecting the webhook implementation", "list the full webhook-file commit history"),
            ("retry-predecessors", "An auditor wants the sequence of webhook changes, not one selected patch", "retrieve the commit log for webhooks.py"),
            ("webhook-shas", "Release analysis needs the hashes and subjects of all webhook-file revisions", "list commits that changed webhooks.py"),
        ],
        "history.request_clarification": [
            ("vague-history", "A request says only 'check the history' without a file or historical question", "ask which file or symbol and revision question matter"),
            ("missing-revisions", "A teammate asks for an old change but names neither target nor revision range", "obtain the missing historical target and bounds"),
            ("unbounded-git", "An audit ticket says 'find what happened before' with no code target", "clarify the code location and historical question before using Git"),
        ],
    },
}

# For file-inspection questions, use evidence tied to the particular scenario
# rather than repeating the pilot's marker for every question about a file.
REPOSITORY_EVIDENCE = {
    "repo.read_service": ["class InsufficientStock", "return stock.get(sku, 0)", "def release_stock"],
    "repo.read_pricing": ["ROUND_HALF_UP", "Decimal(str(tax_rate))", "total.quantize"],
    "repo.read_api": ["def create_order", "def health", '"service": "inventory"'],
    "repo.read_config": ['os.getenv("STOCK_ALERT_LIMIT", "10")', "STOCK_ALERT_LIMIT must not be negative", 'os.getenv("STOCK_ALERT_LIMIT", "10")'],
    "repo.read_webhooks": ["for attempt in range(1, attempts + 1)", "raise DeliveryFailed", "if normalized_id in processed_event_ids"],
    "repo.read_cli": ['separators=(",", ":")', 'return f"{sku}: {quantity}"', "unsupported output format"],
    "repo.read_readme": ["python -m unittest discover -s tests -v", "CopperLedger is a deliberately small Python inventory service", "no third-party runtime dependencies"],
    "repo.read_service_tests": ["test_reserve_stock_rejects_shortage", "test_available_stock_defaults_to_zero", "test_release_stock_returns_units"],
}


def prompt(domain: str, context: str, objective: str, variant: int, action: str) -> str:
    context = context.rstrip(". ")
    objective = objective.rstrip(". ")
    if action.endswith("request_clarification"):
        forms = (
            f"{context}. Select one next action from the catalog before proceeding.",
            f"A teammate reports: {context[0].lower() + context[1:]}. Which single next action is appropriate?",
            f"Situation: {context[0].lower() + context[1:]}. Choose one available action to handle this request.",
        )
    elif domain == "repository_inspection":
        forms = (
            f"{context}. Choose one read-only next action to {objective}.",
            f"A teammate is investigating this case: {context[0].lower() + context[1:]}. Which single read-only action best lets them {objective}?",
            f"Before changing files, consider this report: {context[0].lower() + context[1:]}. Select one read-only action that can {objective}.",
        )
    elif domain == "test_selection":
        forms = (
            f"{context}. Select the one focused existing test command that would {objective}.",
            f"A developer reports: {context[0].lower() + context[1:]}. Which single test command should they run to {objective}?",
            f"Reported case: {context[0].lower() + context[1:]}. Choose one existing test command that can {objective}.",
        )
    else:
        forms = (
            f"{context}. Select one read-only Git action to {objective}.",
            f"For an audit trail, {context[0].lower() + context[1:]}. Which available Git action would {objective}?",
            f"Before citing the history, {context[0].lower() + context[1:]}. Choose one Git action that can {objective}.",
        )
    return forms[variant]


def build() -> dict:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    exemplar = {task["gold_action_id"]: task for task in pilot["tasks"]}
    prefixes = {"repository_inspection": "RI", "test_selection": "TS", "version_history": "VH"}
    tasks = []
    for domain, by_action in SCENARIOS.items():
        assert len(by_action) == 10
        actions = list(by_action)
        for phase in range(3):
            split = "calibration" if phase == 0 else "held_out"
            start = (11, 41, 71)[phase]
            for variant in range(3):
                for action_index, action in enumerate(actions):
                    slug, context, objective = by_action[action][phase]
                    base = exemplar[action]
                    task = {
                        "task_id": f"{prefixes[domain]}-{start + variant * 10 + action_index:02d}",
                        "split": split,
                        "domain": domain,
                        "scenario_family_id": f"{prefixes[domain].lower()}-{slug}",
                        "task_text": prompt(domain, context, objective, variant, action),
                        "repository_context": base["repository_context"],
                        "gold_action_id": action,
                        "wrong_competitor_action_id": base["wrong_competitor_action_id"],
                        "label_rationale": f"Proposed label: {action} is intended to {objective}; {base['wrong_competitor_action_id']} is the frozen near-neighbor control. The researcher must assess semantic uniqueness independently of the mechanical evidence predicate.",
                        "label_evidence": base["label_evidence"],
                        "review_status": "mechanically_verified_pending_researcher_review",
                    }
                    if action in REPOSITORY_EVIDENCE:
                        task["label_evidence"] = {"match_kind": "literal", "marker": REPOSITORY_EVIDENCE[action][phase]}
                    if "mutation" in base:
                        task["mutation"] = base["mutation"]
                    tasks.append(task)
    assert len(tasks) == 270
    assert len({task["task_id"] for task in tasks}) == 270
    assert len({task["task_text"] for task in tasks}) == 270
    assert len({task["scenario_family_id"] for task in tasks}) == 90
    return {
        "schema_version": 1,
        "dataset_id": "jev-tool-alias-main-v1",
        "status": "ai_drafted_mechanically_verifiable_pending_researcher_review",
        "fixture": pilot["fixture"],
        "construction": {
            "source_provenance": "Synthetic scenarios authored for the existing CopperLedger fixture; no public issue text or repository content was adapted.",
            "family_policy": "Three related task questions share one scenario family and occur within one split. Pilot contributes 30 additional families; main contributes 30 calibration and 60 held-out families.",
            "label_policy": "Gold actions and fixed wrong competitors are inherited from the prespecified pilot registry. Mechanical evidence and test mutations are inherited by canonical action; semantic applicability and ambiguity remain pending human review.",
            "paid_model_results_seen_during_construction": False,
        },
        "tasks": tasks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the checked-in dataset and any review decisions")
    args = parser.parse_args()
    if not args.force and (OUTPUT.exists() or REVIEW.exists()):
        parser.error("dataset or review sheet already exists; use --force only before any researcher annotations or model run")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = build()
    OUTPUT.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with REVIEW.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["task_id", "split", "domain", "scenario_family_id", "task_text", "gold_action_id", "wrong_competitor_action_id", "label_rationale", "label_evidence", "review_decision", "reviewer", "review_date", "review_notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in dataset["tasks"]:
            writer.writerow({
                **{key: task[key] for key in fields if key in task},
                "label_evidence": json.dumps(task["label_evidence"], ensure_ascii=False),
            })
    print(OUTPUT)
    print(REVIEW)
