# Tool Alias Bias Study

This repository contains the public research package for a controlled study of
how exact visible aliases for an unchanged backend action affect automatic
acceptance under thresholds fitted on an unaliased tool catalog.

## Main observed result

For Jev's prespecified P2 comparison, changing from C0 (one visible option per
canonical action) to C2 (three exact aliases of the proposed correct action)
changed automatic coverage by **-77.81 percentage points** (95% family-cluster
bootstrap interval: -85.96 to -69.10; 178/180 retained held-out tasks; 60
scenario families). The canonical backend action changed on 10/356 paired
requests. This is an automatic-coverage result under a frozen cutoff, not an
equally sized action-accuracy loss.

## Evidence status

- The observed matrix is **partial**: 10,800 planned logical requests produced
  10,787 valid final distributions and 13 invalid final responses.
- All 300 proposed labels were AI drafted and mechanically checked, but none
  has completed independent blind researcher review. Correctness-dependent
  outcomes are therefore provisional.
- The raw provider outputs are a separate release asset so that the Git
  repository stays small and cloneable.

## Repository contents

| Path | Purpose |
|---|---|
| `study/` | Three-arm study runner, analysis, protocol, and frozen 300-task data |
| `pilot/` | Registry, adapters, execution checks, and fixture-history bundle |
| `results/observed-study/` | Compact, versioned outputs from the recorded study |
| `scripts/` | Fixture restoration and release verification tools |
| `tests/` | Offline structural checks for the public package |
| `docs/` | Reproduction and release instructions |

## Verify the recorded study offline

Requirements: Python 3.11+ and Git on `PATH`. No API key is needed.

```powershell
python scripts/restore_fixture.py
python scripts/verify_release.py
```

To regenerate the analysis from the raw-data release asset, download the asset
named in `results/observed-study/raw-data.md`, extract it at the repository
root, then run:

```powershell
tar -xzf tool-alias-bias-observed-study-v1-raw.tar.gz
python -m study.run --analyze-only --out results/observed-study/raw
```

The command makes no network request and does not require credentials. It writes
recomputed analysis files inside the extracted raw-data directory. Compare the
primary endpoint, thresholds, and accounting with the compact files in
`results/observed-study/`.

## Run a new experiment

Restore the fixture first, set an OpenRouter API key in your own environment,
and run the study under a new output directory:

```powershell
$env:OPENROUTER_API_KEY = "your-key-here"
python -m study.run --out results/my-run
```

The default local planning guard is USD 10; it is not a provider-side billing
cap. A new run can differ because model versions, providers, and routing may
change. Do not overwrite `results/observed-study/`.

## Citation and licenses

Use `CITATION.cff` for citation metadata. Code is released under MIT; the
synthetic benchmark inputs and observed-study outputs are released under CC BY
4.0, subject to the provider-output notice in `DATA-LICENSE.md`.
