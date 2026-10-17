# Pilot task-label review sheet

Review this sheet before any model output is examined. The correct label is a **canonical backend action**, not a visible `tool_XX` name. Mark each item approve, replace, or ambiguous; record disagreements rather than repairing labels after a model run.

The current `mechanically_verified_pending_researcher_review` status means the runner has checked a unique fixture-output predicate for the label and ruled out the C5 controls; the nine concrete test labels additionally have a task-specific mutation that fails only their labeled executable test candidate. This is still not independent human review. After sign-off, set the corresponding JSON `review_status` to `researcher_approved` and preserve this sheet with the run artifacts.

| ID | Task focus | Gold canonical action | Frozen wrong competitor | Mechanical evidence | Reviewer decision / initials / date |
|---|---|---|---|---|---|
| RI-01 | Shortage exception path | `repo.read_service` | `repo.read_api` | `class InsufficientStock` | |
| RI-02 | Half-cent tax rounding | `repo.read_pricing` | `repo.read_service` | `ROUND_HALF_UP` | |
| RI-03 | Health response payload | `repo.read_api` | `repo.read_config` | `def health` | |
| RI-04 | Deployed alert limit | `repo.read_config` | `repo.read_readme` | `STOCK_ALERT_LIMIT` | |
| RI-05 | Restock retry behavior | `repo.read_webhooks` | `repo.read_service` | `def deliver_restock_notice` | |
| RI-06 | Compact JSON formatter | `repo.read_cli` | `repo.read_api` | `json.dumps` | |
| RI-07 | Documented test command | `repo.read_readme` | `repo.read_service_tests` | README purpose sentence | |
| RI-08 | Reservation test contract | `repo.read_service_tests` | `repo.read_service` | shortage test name | |
| RI-09 | `reserve_stock` call sites | `repo.search_reserve_stock` | `repo.read_service` | source-search result | |
| RI-10 | Unbounded inspection request | `repo.request_clarification` | `repo.read_service` | clarification result | |
| TS-01 | Shortage guard | `test.service_unit` | `test.checkout_integration` | shortage test + isolated guard mutation | |
| TS-02 | Commercial rounding | `test.pricing_unit` | `test.checkout_integration` | half-up test + isolated rounding mutation | |
| TS-03 | Health response | `test.api_unit` | `test.service_unit` | health test + isolated response mutation | |
| TS-04 | Delivery retry | `test.webhooks_unit` | `test.webhook_dedup_integration` | retry test + isolated loop mutation | |
| TS-05 | JSON formatting | `test.cli_unit` | `test.api_unit` | JSON test + isolated formatter mutation | |
| TS-06 | Hold plus tax | `test.checkout_integration` | `test.pricing_unit` | checkout test + isolated return mutation | |
| TS-07 | Restock-to-availability handoff | `test.restock_integration` | `test.webhooks_unit` | handoff test + isolated availability mutation | |
| TS-08 | Duplicate restock event | `test.webhook_dedup_integration` | `test.restock_integration` | duplicate-event test + isolated dedup mutation | |
| TS-09 | Zero-unit release regression | `test.reservation_regression` | `test.service_unit` | regression test + isolated release mutation | |
| TS-10 | Unbounded test request | `test.request_clarification` | `test.service_unit` | clarification result | |
| VH-01 | Reservation introduction patch | `history.show_reserve_introduction` | `history.blame_service` | annotated tag | |
| VH-02 | Price-rounding patch | `history.show_price_rounding` | `history.log_webhooks` | annotated tag | |
| VH-03 | Webhook-retry patch | `history.show_webhook_retry` | `history.show_config_default` | annotated tag | |
| VH-04 | API-health patch | `history.show_api_health` | `history.show_cli_json` | annotated tag | |
| VH-05 | CLI JSON patch | `history.show_cli_json` | `history.show_api_health` | annotated tag | |
| VH-06 | Alert-limit patch | `history.show_config_default` | `history.show_api_health` | annotated tag | |
| VH-07 | Regression before/after diff | `history.diff_reservation_regression` | `history.show_reserve_introduction` | added validation line | |
| VH-08 | Current definition attribution | `history.blame_service` | `history.show_reserve_introduction` | blame-line pattern | |
| VH-09 | All webhook commits | `history.log_webhooks` | `history.show_webhook_retry` | full-hash/tab log row | |
| VH-10 | Unbounded history request | `history.request_clarification` | `history.log_webhooks` | clarification result | |

## Reviewer notes

- C5 correction already applied: the version-history menu uses `history.log_security` as its third load control. It replaced `history.blame_config`, which a separate audit found could reasonably answer VH-06. Recheck semantic inapplicability rather than relying only on the mechanical output predicates.
- Are any task phrases underspecified enough that another action is equally justified?
- Does the frozen wrong competitor represent a credible near-neighbor rather than an artificial distractor?
- Does every C5 control remain genuinely distinct and inapplicable to the stated task?
- For the three clarification tasks, is essential information truly absent?
- If a task is replaced before live calls, record the reason, preserve the superseded version, rerun `python -m pilot.verify`, and do not reuse its output as a held-out result.
