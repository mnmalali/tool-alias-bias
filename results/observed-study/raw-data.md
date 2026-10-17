# Raw-data release asset

The full observed-study record is a separate GitHub release asset because the
physical-attempt ledger alone is larger than GitHub's ordinary Git file limit.
It contains 18 study artifacts under `results/observed-study/raw/`, including
the plan, pre-dispatch intents, captured requests and responses, attempt ledger,
preflight evidence, raw paired outputs, and accounting tables. An internal
drafting handoff note is deliberately excluded.

**Release-asset filename:** `tool-alias-bias-observed-study-v1-raw.tar.gz`  
**SHA-256:** `f683d77aef5f6b6e06f684003a90eed8777a5f6a0efa7adac1d4e733355c6ff2`  
**Download URL:** https://github.com/mnmalali/tool-alias-bias/releases/download/v0.1.0/tool-alias-bias-observed-study-v1-raw.tar.gz

## Extract and verify

From the repository root:

```powershell
tar -xzf tool-alias-bias-observed-study-v1-raw.tar.gz
python scripts/verify_release.py --raw-dir results/observed-study/raw
python -m study.run --analyze-only --out results/observed-study/raw
```

The analysis command is offline: it reads the retained ledgers and writes fresh
derived output in the extracted raw directory. It does not send a provider
request or require an API key.

## Data boundary

The archive records a partial matrix and unreviewed proposed labels. It includes
provider responses routed through OpenRouter, TypeSafe AI, OpenAI, and
Anthropic/Google. Review the applicable provider terms before redistributing the
release asset. The tracked `release-metadata.json` records the expected raw
ledger hash and asset checksum.
