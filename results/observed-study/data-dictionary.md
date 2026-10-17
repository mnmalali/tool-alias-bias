# Data dictionary

## `manifest.json`

This is the run manifest produced before and during the observed matrix. Its
`dataset_hash` and `registry_hash` use the harness's canonical JSON
serialization, not the byte-level file hash. Byte-level SHA-256 values for the
published task and registry files are in `release-metadata.json`.

## `study-results.json`

This file records completeness, cost accounting, the prespecified Jev primary
endpoint, the Luna secondary endpoint, threshold-fitting accounting, and stated
limitations. `status: partial` means that 13 final responses were invalid even
though all 10,800 logical requests have final rows.

## `thresholds.json`

For each model and policy, `cutoff` is the frozen C0-calibration gate. A score
is accepted when `score >= cutoff`. These cutoffs were fitted before held-out
condition comparison.

## `paired-estimates.json`

Each record compares one catalog condition with C0 for a model and policy. It
reports coverage, accepted-correct, accepted-wrong, action-change, and
accept/defer contrasts when the complete-pair data permit them. Intervals use
2,000 deterministic family-cluster bootstrap replicates.

## `condition-metrics.csv`

One row per split, model arm, and catalog condition. Columns give planned and
final logical request counts, valid distributions, failed/invalid finals,
physical attempts, retries, token totals, known costs, unknown-cost attempts,
and physical latency summaries.

## `complete-pair-tasks.csv` and `tie-rates.csv`

`complete-pair-tasks.csv` contains held-out tasks with two valid repeats in C0
and the comparison condition. `tie-rates.csv` contains exact visible-option and
action-mass tie diagnostics by split, model, and condition.
