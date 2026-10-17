# Main-study task set

`main_tasks.json` is the fixed, synthetic extension of the 30-task CopperLedger
pilot. It adds 90 calibration and 180 held-out prompts, equally allocated to
repository inspection, test selection, and version-history inspection. The
combined study has 300 prompts. The main set has 90 scenario families with
three related prompts each; the pilot has 30 further families. The held-out
sample therefore has 60 scenario families, not 180 independent incidents.

The scenarios were authored against the local fixture and the existing
10-action-per-domain registry. No public issue text was adapted. Each proposed
gold action and frozen wrong competitor were set before paid responses. The
`label_evidence` predicate is an executable check of a candidate action's
output. The nine concrete test actions retain the pilot's isolated mutation
check. Neither check establishes that a human would find every prompt
unambiguous; `review_status` remains
`mechanically_verified_pending_researcher_review` until a researcher checks
`main_label_review.csv` without looking at model answers.

`build_main_tasks.py` records the source scenario seeds and can reproduce the
JSON and CSV. It refuses to overwrite either file by default, protecting any
review decisions or frozen run inputs. `--force` replaces them and should only
be used before human annotation or a model run. Once an experiment starts,
archive the exact dataset hash in the run manifest and do not regenerate it.

This benchmark tests routing for a small synthetic repository and fixed
candidate actions. Repeated prompt styles and shared action targets limit
generalization. The analysis must resample scenario families, preserve paired
conditions and repeats, and report the 60-family held-out count alongside the
180-prompt count.
