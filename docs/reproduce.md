# Reproducing the observed study

## Offline verification

1. Install Python 3.11 or newer and Git.
2. Clone the repository and open a terminal at its root.
3. Run `python scripts/restore_fixture.py` to reconstruct the synthetic
   CopperLedger fixture and its required Git history.
4. Run `python scripts/verify_release.py` to verify the tracked benchmark and
   compact result files.
5. Download the raw-data release asset described in
   `results/observed-study/raw-data.md` and extract it at the repository root.
6. Run `python scripts/verify_release.py --raw-dir results/observed-study/raw`.
7. Run `python -m study.run --analyze-only --out results/observed-study/raw`.

Steps 1–7 make no network request to a model provider. The last command
regenerates analysis output in the extracted raw-data directory. Its primary
endpoint, secondary endpoint, and thresholds should equal the tracked compact
output files.

## Fresh experiment

`python -m study.run --out results/my-run` runs a new three-arm matrix only
when `OPENROUTER_API_KEY` is set. This is a new paid experiment; it is not a
method of reproducing the recorded provider responses. Read `study/PROTOCOL.md`
before running it and keep the observed-study directory unchanged.
