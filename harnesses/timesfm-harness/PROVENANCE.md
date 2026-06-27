# timesfm-harness — provenance

This directory is a **metaharness bundle** (ADR-041 "metaharness as program
synthesis") generated for the `crates/timesfm` + `crates/ruvector-timesfm`
TimesFM forecasting work, using the `agent-harness-generator` at
`ruvnet/agent-harness-generator`. It is the authentic generator output, not
hand-authored.

## How it was generated (reproducible)

Generator: `create-agent-harness` CLI **v0.2.7**, kernel `0.1.2`,
binary `packages/create-agent-harness/dist/bin.js`.

```bash
cd /path/to/agent-harness-generator
BIN=packages/create-agent-harness/dist/bin.js

# Feasibility scorecard + repo genome (read-only analysis of the crate)
node $BIN score  /path/to/ruvector/crates/timesfm --json
node $BIN genome /path/to/ruvector/crates/timesfm --json

# Synthesize the bundle (engineering-pod vertical, Claude Code host)
node $BIN timesfm-harness \
  --template vertical:coding --host claude-code \
  --description "TimesFM 1.0 200M decoder-only time-series forecasting inference crate (Rust/candle) — engineering pod harness" \
  --target <output-dir>
```

`vertical:coding` (engineering pod: architect / implementer / reviewer /
test-writer over code memory) was the generator's own recommended template for
this `rust-crate-harness` archetype — the right fit for developing a Rust
inference library.

## Score (feasibility scorecard)

```json
{ "schema":1, "repo":"timesfm", "harnessFit":52, "compileConfidence":90,
  "taskCoverage":79, "toolSafety":100, "memoryUsefulness":34,
  "estCostPerRunUsd":0.048, "recommendedMode":"CLI + MCP",
  "archetype":"rust-crate-harness", "template":"vertical:coding",
  "scaffoldReady":true, "hardConstraints":"6/6" }
```

## Genome (repo synthesis verdict)

```json
{ "repo_type":"rust", "agent_topology":["maintainer","tester","security"],
  "risk_score":0.37, "mcp_surface":"local_default_deny",
  "test_confidence":0.5, "publish_readiness":0.55 }
```

## Witness / provenance (ADR-011)

`.harness/manifest.json` records `schema:1`, the template/vars/hosts, and a
SHA-256 for every emitted file. `.harness/manifest.sha256` is the witness over
the manifest:

```
manifest witness = 7c45ab915393da6e43141935ce884d1718b3dcabaf2454f3fe32519999e32a7c
```

Verify integrity:

```bash
sha256sum .harness/manifest.json   # must equal the contents of .harness/manifest.sha256
```

Verified valid at commit time.

## Connection to the RuVector TimesFM work

The harness governs an engineering pod for the forecasting crates. The
runtime forecasting capability it would orchestrate is the
`time_series_forecast` MCP tool, implemented by the
`ruvector-timesfm-forecast` CLI in `crates/ruvector-timesfm` (JSON in →
point + p10/p50/p90 out).

## Optimizing the harness — Darwin evolve via OpenRouter (key from GCP)

The harness ships a Darwin-Mode self-improvement loop (the `evolve` skill). Its
default mutator is deterministic (air-gapped, no key). The **LLM mutator**
(`OpenRouterMutator`, ADR-071) is library-only — not exposed by the
`metaharness-darwin` CLI — so `scripts/evolve-openrouter.{sh,mjs}` wire it into
the `evolve()` engine.

The OpenRouter API key is **sourced from GCP Secret Manager at runtime** and
exported only into the run's process — never stored in the repo, a dotfile, or
the logs:

```bash
# real sandbox, key fetched from GCP secret OPENROUTER_API_KEY (project cognitum-20260110)
./scripts/evolve-openrouter.sh
# tune cost/scope:
GENERATIONS=1 CHILDREN=2 SANDBOX=mock ./scripts/evolve-openrouter.sh
# overrides: OPENROUTER_SECRET, GCP_PROJECT, DARWIN_MUTATOR_MODEL, DARWIN_DIST
```

Validated run (real sandbox, 1 gen × 2 children, `google/gemini-2.5-flash`):
baseline scored **0.985** (taskSuccess 1.0, testPassRate 1.0, safety 1.0, zero
secret-exposure/destructive/hallucination flags); 2 real OpenRouter mutations,
~$0.003. Every mutation passes the `validateGeneratedCode` safety gate (no new
imports/network/shell/env) and only promotes on measured improvement.

## Notes on this generator version (v0.2.7)

- `mint` emits `.harness/manifest.json` + `.harness/manifest.sha256` (the
  witness). The MCP policy is embedded in `.claude/settings.json`
  (default-deny). It does **not** emit separate `.harness/genome.json` /
  `.harness/mcp-policy.json` files — `genome` is a read-only command (captured
  above) and the policy lives in settings.json.
