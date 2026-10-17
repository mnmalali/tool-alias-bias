# Jev tool-alias study: pre-run execution specification

**Dated:** 2026-10-09 (Asia/Dubai)  
**Comparator amendment:** 2026-10-11 (Asia/Dubai), before any live model calls; the user replaced Haiku 4.5 with Haiku 5.5.  
**Three-arm amendment:** 2026-10-12 (Asia/Dubai), before any live model calls; the user added GPT-6 Luna Decisions. Jev P2 C2–C0 remains the sole primary endpoint.  
**Compatibility correction:** 2026-10-12 (Asia/Dubai), after a five-attempt smoke and before any matrix call. The pinned Haiku endpoint rejected the request during parameter routing; the corrected request omits unsupported `temperature`. No outcome or threshold was changed. A fresh six-call smoke is required.  
**Endpoint amendment:** 2026-10-12 (Asia/Dubai), after a second six-attempt smoke and before any matrix call. Haiku C0 passed on Anthropic; Haiku C2 timed out at 90 seconds with no HTTP response. The request now pins Google Vertex `google-vertex/global`, whose listing supports the required parameters at the same posted price. A fresh smoke must confirm it; no outcome or threshold changed.  
**Schema compatibility amendment:** 2026-10-12 (Asia/Dubai), after a third six-attempt smoke and before any matrix call. Google Vertex passed Haiku C0 but Haiku C2 timed out at about 92.5 seconds, repeating the C2 failure across two providers. Canonicalized the strict Haiku schema's option-key order and removed unsupported numeric bounds; the same full keyed vector and local numeric checks remain. Haiku C2 is now the first smoke request. These are compatibility changes, not model results, and require a new credentialed smoke.  
**Validator correction:** 2026-10-12 (Asia/Dubai), after a fourth smoke's first request returned a complete Haiku C2 vector summing to decimal 0.99. The already intended inclusive ±0.01 rounding allowance was implemented with binary floating-point comparison and rejected this boundary value. The validator now compares the decimal sum; raw probabilities are retained without repair or renormalization. No endpoint, dataset, or outcome changed.  
**Transport retry amendment:** 2026-10-12 (Asia/Dubai), after a fifth smoke passed Haiku C2 but Haiku C0 timed out at 90 seconds without an HTTP response. The smoke now applies the same bounded retry count as the matrix only to retryable transport/rate errors. Every physical attempt has a unique fsynced intent, response ledger row, and known cost or unpriced reserve. Completed invalid model answers are not retried. Six successful logical probes remain required before the matrix.  
**Client-timeout amendment:** 2026-10-12 (Asia/Dubai), before any matrix call. The fifth smoke's Haiku C0 request was cut off by the local 90-second client without an HTTP response. The shared smoke/matrix default is now 210 seconds per request, above Anthropic's documented 180-second schema-compilation timeout with some transport margin. Compilation is a possible explanation, not an established cause of this request's timeout. The CLI still allows `--timeout-seconds N`; retry count and cost guard are unchanged.  
**Haiku measurement amendment:** 2026-10-12 (Asia/Dubai), after a sixth smoke and before any matrix call. A retried Haiku C2 response contained all 13 finite nonnegative values but summed to 0.98; an earlier response summed to 0.99. Repeated near-unit failures show that treating a chat model's elicited numbers as exact native probabilities makes the smoke gate fragile. Haiku now reports nonnegative relative weights; the adapter divides each by their positive total, retaining the original values and sum in the ledger. Jev and Luna native vectors keep their existing strict unit-sum validation. This amendment affects Haiku measurement and its exploratory comparisons, not the fixed Jev primary endpoint, dataset, or cutoffs. The change was made after compatibility observations, so the protocol is not a preregistration. A fresh live smoke remains required.  
**Status:** Fixed local design and analysis code; six credentialed smokes stopped before the matrix, and no
human label approval has occurred. This file is a project protocol, not a
registered preregistration or an empirical result.

## Research question and contribution boundary

For the bounded CopperLedger dispatch benchmark, how does adding exact visible
aliases for an unchanged backend action affect selected actions and the
automatic-decision coverage of cutoffs fitted on the original catalog in two
native decision models, Jev and GPT-6 Luna Decisions? Does action-level
probability aggregation maintain useful coverage without increasing wrong
automatic actions? Claude Haiku 5.5 is a contextual chat-model comparator
using an elicited probability vector.

Equivalent-tool bias and option-order effects already have direct prior art,
including Faghih et al., *Tool Preferences in Agentic LLMs are Unreliable*
([EMNLP 2025 paper](https://arxiv.org/html/2505.18135)). The proposed
increment is the measured transfer of a **frozen abstention gate** for Jev's
native choice probabilities under exact same-backend alias edits, with a
duplicated-wrong-action control. No firstness claim is justified. The fixed
registry and fixture make this a local microbenchmark, not evidence of
end-to-end coding-agent productivity or general safety. Existing Jev–Luna
comparisons benchmark task accuracy or cost; this study tests exact-backend
alias perturbations and frozen accept/defer gates on a bounded fixture.

## Inputs fixed before model output

`study/data/study_tasks.json` combines 30 pilot, 90 C0-threshold calibration,
and 180 held-out task prompts. Each split is evenly divided among repository
inspection, test selection, and version-history inspection. The combined
dataset has 120 scenario families: 30 pilot, 30 calibration, and 60 held-out.
Three related prompts share each main-study family. Scenario families stay in
one split. The held-out statistical sample therefore has **60 families** (20
per domain), although it contains 180 task prompts and 6,480 model requests.

The tasks and fixture are synthetic. The pilot tasks, registry, scenarios,
fixed wrong competitors, and main-study generation code are retained in the
repository. Gold labels are canonical executable actions, not visible tool
IDs. Exact alias equivalence is established by registry contract comparison
and dispatch-output checks. Label-evidence predicates and focused test
mutations mechanically check candidate targets. They do not establish human
semantic agreement. The `review_status` and review sheets retain that
distinction; a later researcher review should be completed without consulting
model choices and should never silently relabel held-out examples after
results are seen.

The actual dataset and source hashes, action mapping, displayed catalog,
ordered request body, model settings, and run-order seed are stored in the run
manifest and ledger. A resumed run must match them. The dataset generator
refuses to overwrite its checked-in output without an explicit `--force`.

## Intervention and request matrix

| Catalog | Change relative to C0 |
| --- | --- |
| C0 | One visible option per canonical action. |
| C1 | One extra alias of the gold action. |
| C2 | Three extra aliases of the gold action. |
| C3 | One extra alias of a prespecified wrong competitor. |
| C4 | Three extra aliases of that wrong competitor. |
| C5 | Three distinct, inapplicable control actions. |

An alias copies the backend identity, bound arguments, schema, permissions,
preconditions, observable effect, and description. Neutral equal-length IDs
and deterministic position schedules are used; the question and repository
context are unchanged within a task. The C5 control approximately matches
catalog load, but cannot perfectly isolate token length from semantics.

Each task and condition receives two identical requests from each model. The
planned matrix is `300 × 6 × 2 × 3 = 10,800` logical model requests: 1,080 pilot,
3,240 calibration, and 6,480 held-out. Repeats measure service variability;
they are not independent tasks. Conditions and models are interleaved in a
fixed randomized order. Failed transport attempts and separate live smoke
requests are additional physical requests and must remain in cost accounting.

The pinned arms are OpenRouter `typesafe/jev-1.13` and
`openai/gpt-6-luna-decisions` through its Decisions API, plus
`anthropic/claude-haiku-5.5` through Chat Completions, with Haiku extended
thinking disabled and its serving endpoint constrained to OpenRouter's
`google-vertex/global` route. Jev and Luna option vectors are native API outputs. Haiku returns a structured JSON vector of relative weights, which is divided by its positive total before P2/P3/P4 analysis. Original weights and their sum are retained. This transformed elicitation is a different measurement interface; between-model calibration or Brier-score differences cannot be interpreted as an intrinsic ranking of native probability quality. The returned model/provider/snapshot,
timestamps, usage, cost, response body, failures, and latency are retained.
The API key belongs in `OPENROUTER_API_KEY` and is excluded from artifacts.

## Policies and threshold selection

For each valid response, let `p(o)` be the native probability of visible option `o` for Jev and Luna, or Haiku's normalized elicited weight,
and let `g(o)` be its exact registry backend action. Aggregate
`q(a) = sum_{o:g(o)=a} p(o)` only across options proven to share that action.

| Policy | Chosen action | Gate score |
| --- | --- | --- |
| P1 (native decision diagnostic) | Backend of the native chosen option for Jev or Luna | Native `confidence` |
| P2 (primary) | Backend of the maximum-probability visible option | Maximum `p(o)` |
| P3 | Maximum-mass backend action | Maximum `q(a)` |
| P4 (engineering control) | P2 after exact-alias removal | Linked C0 score and action |

P4 reuses the paired C0 response: it is a deterministic canonical-request
control, not extra inference or a learned improvement. Native
`confidence` is a ranking/gating statistic, not presumed to be a calibrated
probability of correctness. P2 ties use lowest neutral visible ID; P3 ties use
lowest neutral canonical base ID.

After the run, fit separate Jev P1/P2, Luna P1/P2, and Haiku P2 cutoffs using **only
C0 calibration responses** (90 prompts × two repeats = 180 planned responses
per model). In a partial run, a calibration task contributes only if both C0
repeats are valid; require at least 80 of 90 such tasks **for each model** or
withhold threshold-based results. Record retained counts and attained
coverage. The accepted-score rule is `score >= cutoff`. Choose the attainable
coverage nearest 80%; if equally close, choose lower coverage. P3 and P4
share that model's P2 cutoff because C0 has one label per action. Do not refit
on any intervention or held-out outcome.

## Prespecified outcomes and interpretation

**Primary endpoint:** on held-out Jev P2 requests, the paired signed coverage
change `coverage(C2) - coverage(C0)` under the frozen C0 cutoff. Luna's same
contrast is a prespecified secondary replication; Jev–Luna differences are
exploratory because the services can differ in implementation and routing. The
predeclared practically meaningful shift is 10 percentage points. Report the
estimate and 95% family-cluster bootstrap interval regardless of sign. An
interval crossing zero is inconclusive; absence of an observed shift does
not by itself prove equivalence.

For every model, policy, and catalog report unconditional canonical-action
accuracy, automatic coverage, selective risk (wrong among accepted; undefined
when none are accepted), accepted-wrong rate (wrong accepted among all),
accepted-correct yield, action and accept/defer changes versus C0, identical
repeat disagreement, latency, failures, tokens, and billed cost. Report C3/C4
wrong-action duplication even if it weakens the proposed mitigation. Compare
P3 with P2 using absolute coverage deviation from C0, accepted-correct yield,
and accepted-wrong rate. Since `max q(a) >= max p(o)` on a valid distribution,
P3's extra coverage alone is an algebraic consequence and not evidence of
benefit. Action-distribution Brier scores and risk-coverage curves are
secondary diagnostics.

For intervals, average the two repeats within each held-out task, then
resample entire scenario families **within domain**, preserving all prompts,
conditions, model arms, and repeats together. Use 2,000 deterministic
bootstrap replicates with seed 20261009. Report the 60-family sample size.
Domain strata have only 20 families each and should be treated as exploratory.

## Failures, review, and reproducibility

The live command first requires six separately logged API smoke probes
(C0 and C2 on each arm) and stops if request or response formats are
incompatible. Retryable smoke transport/rate errors may produce additional
physical calls within the configured retry bound; their intents, results,
costs or unknown-cost reserves are logged separately. The matrix writes and
syncs an intent before dispatch; resume
refuses an intent lacking a ledger acknowledgement because a process crash
could leave a billed request with unknown outcome. Observed physical attempts
and bounded transport retries are retained. A successful HTTP response with
malformed native probabilities or unusable Haiku weights is retained, not silently repaired or retried as a
new semantic answer. Fatal HTTP 400/401/402/403/404/413/422 errors stop the
matrix, as do three consecutive final invalid responses in one model arm.
Jev and Luna native vectors must have exactly the visible option IDs, finite
values in [0,1], and a decimal sum within an inclusive ±0.01 rounding
allowance; their values are never renormalized. Haiku must provide every visible option ID with finite relative weights in [0,1], a valid selected option and confidence, and a strictly positive total weight. Its values are divided by that total; the reported vector, sum, and transform name remain in the ledger and summary diagnostics. A missing key, invalid number, or zero total remains a failed response. Haiku risk-coverage and Brier diagnostics describe this normalized elicitation only.

If all 10,800 planned final responses are valid, use the complete planned
analysis. For any missing or invalid final response, use the **prespecified
paired complete-case analysis**: a held-out task contributes to a given
condition-versus-C0 contrast only when both repeats of C0 and that condition
are valid for that model. Average within task, then resample retained scenario
families within domain. Report retained tasks and families for every contrast,
failure counts by split/model/condition, and the complete-case estimate as
conditional on observed pairs. For the primary Jev P2 C2–C0 endpoint, also
report a conservative full-180-task interval that assigns each excluded task
any signed coverage change from -1 to +1. These bounds address missing
held-out outcomes, not bias in the threshold caused by excluded calibration
tasks. If fewer than 80 calibration tasks per model survive, or no primary
paired tasks survive, withhold a threshold-based endpoint. Do not turn an
invalid response into an abstention or zero probability.

The local `--max-cost-usd` guard reserves forecast cost and stops further
dispatches if the limit would be exceeded. It is **not** a provider-side hard
billing cap; an already issued request can cost more than forecast. The
working three-model estimate is approximately $2.01–2.28 for one 10,800-call
matrix under 2,000 input tokens/request and 150–300 Haiku output tokens/request,
using the October 12 OpenRouter list prices. This includes $0.3024 for Jev,
$0.72 for Luna, and $0.99–1.26 for Haiku. Higher actual
spend possible from longer prompts, smoke calls, retries, or provider changes.
The code does not spend until the user supplies a key and executes live mode.

The one-command live path permits the existing pending label-review state and
records it in the manifest so data can be gathered with only the user's key.
Such a run measures model behavior, but correctness-based paper claims remain
provisional until an independent researcher checks task ambiguity and labels
without using model outcomes. Any post-run exclusion or correction must be
reported explicitly, never presented as prespecified.
