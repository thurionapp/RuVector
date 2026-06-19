# ADR-260: Darwin Mode as Evolutionary Substrate for MetaHarness

**Status:** Accepted  
**Date:** 2026-06-18  
**Supersedes:** Extends ADR-259 (ruvllm local mutator — point integration)  
**Components:**
- `@metaharness/darwin` (agent-harness-generator) — harness evolution loop
- `crates/ruvllm` — local LLM inference with RDT/OpenMythos (ADR-258)
- `crates/ruvector-core` — HNSW vector index for archive storage
- `npm/packages/ruvector` — TypeScript bindings for ruvvector

---

## Executive Summary

Darwin Mode is **the right evolutionary substrate for MetaHarness**, but the
existing ADR-259 captures only the shallowest integration (ruvllm as a drop-in
mutator). The deeper opportunity is a **three-layer stack**:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Evolution  │  darwin-mode evolve loop             │
│                      │  Mutate → Sandbox → Score → Archive  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Inference  │  ruvllm serve (RDT/OpenMythos/GGUF)  │
│                      │  Local CodeGenerator — zero API cost  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Memory     │  ruvvector HNSW                      │
│                      │  Semantic population archive          │
└─────────────────────────────────────────────────────────────┘
```

ADR-259 wires Layer 2 → Layer 3. This ADR wires Layer 1 → Layer 3 and documents
the emergent properties of the full stack.

---

## Context

### What Darwin Mode Actually Is

Darwin Mode is a **harness-as-genetic-material** system. The model is frozen;
the harness evolves. Each generation it:
1. Selects a parent variant from the population archive
2. Calls a `CodeGenerator` to mutate ONE of 7 approved surfaces
3. Validates the mutant (security gate + sandbox execution)
4. Scores it against the frozen kernel (6-term deterministic scorer)
5. Archives the scored variant into the population tree

This is not a training loop. It is **test-time search** over the harness
configuration space — the same spirit as test-time compute scaling (ADR-258)
but applied to the agent's operating system rather than the model's inference depth.

### What MetaHarness Currently Does

MetaHarness generates a 7-file agent harness from a repo profile. It produces
the initial `planner.ts`, `context_builder.ts`, `reviewer.ts`, `retry_policy.ts`,
`tool_policy.ts`, `memory_policy.ts`, and `score_policy.ts`. The harness is
generated once and deployed.

**The gap:** generated harnesses are not improved after deployment. Darwin Mode
closes this gap by turning the harness into a live artifact that improves over
generations.

### The Population Archive Bottleneck

Darwin Mode's default archive is the **filesystem** (`$repo/.metaharness/`). The
archive stores the full text of each variant's 7 surface files and a JSON score
record. Population selection reads all variants, computes distances, and picks
parents using one of eight strategies (score, quality-diversity, behavioral-
diversity, niche-steering, clade, pareto, whole-archive, random).

For large populations (100+ variants, 10+ generations), three problems emerge:
1. **No semantic search** — quality-diversity and behavioral-diversity strategies
   use cosine distance on 384-dim text embeddings, but compute them fresh every
   selection call (O(n × 384) per generation).
2. **No cross-repo correlation** — each repo evolution is isolated; a successful
   `retry_policy.ts` mutation from one repo never informs another.
3. **No persistence across runs** — archive lives in the repo working tree;
   re-running `metaharness-darwin evolve` after a clean reinitializes the archive.

**ruvvector solves all three.**

---

## Decision

### Integration 1 (ADR-259): ruvllm as CodeGenerator (implemented)

`RuvllmMutator` → `POST /v1/chat/completions` → local RDT/GGUF model.
Zero API cost, air-gap capable, sub-300ms latency. See ADR-259 for details.

### Integration 2 (this ADR): ruvvector as Population Archive

Replace or augment the filesystem archive with a **ruvvector HNSW namespace**.
Each scored variant is indexed as a vector; selection uses ANN search instead of
exhaustive scan.

#### Architecture

```
darwin-mode evolve loop
  │
  ├── scoreVariant(variant) → ScoreRecord
  │
  ├── archive.upsert(variant, score)     ← TODAY: write JSON to .metaharness/
  │                                        PROPOSED: upsert to ruvvector namespace
  │
  └── archive.select(strategy, k=4)     ← TODAY: load all variants, compute pairwise
                                          PROPOSED: ANN query on embedded code surface
```

#### Embedding scheme

Each variant is embedded as the **concatenation of its 7 surface file hashes**
plus a 384-dim ONNX embedding of the `planner.ts` content (the surface with the
highest behavioral variance empirically). This gives a 400-dim behavior descriptor
that is:
- **Fast to compute** (SHA256 × 7 + one ONNX inference)
- **Discriminative** (planner logic drives most behavioral divergence)
- **Portable** (same ruvvector node used for all other vector workloads)

#### TypeScript binding (new `archive-ruvvector.ts`)

```typescript
// packages/darwin-mode/src/archive-ruvvector.ts
// RuvvectorArchive — HNSW-backed population store for Darwin Mode (ADR-260).

import { RuVectorDB } from '@ruvector/ruvector';  // npm/packages/ruvector
import type { ScoredVariant } from './types.js';

const NS = 'darwin-archive';

export class RuvvectorArchive {
  private db: RuVectorDB;

  constructor(dbPath: string = '.ruvvector/darwin.db') {
    this.db = new RuVectorDB({ path: dbPath });
  }

  async upsert(variant: ScoredVariant, embedding: Float32Array): Promise<void> {
    await this.db.upsert(NS, {
      id:       variant.id,
      vector:   embedding,
      metadata: {
        score:      variant.score,
        generation: variant.generation,
        surface:    variant.mutatedSurface,
        parentId:   variant.parentId,
      },
    });
  }

  /** Approximate nearest neighbours — O(log n) vs O(n) for exhaustive. */
  async selectDiverse(k: number, queryEmbedding: Float32Array): Promise<string[]> {
    const results = await this.db.search(NS, queryEmbedding, { limit: k * 4 });
    // Greedy diversity: pick the k most spread apart by score-normalized distance
    return greedySpread(results, k).map(r => r.id);
  }

  async selectByScore(k: number): Promise<string[]> {
    const all = await this.db.list(NS, { sortBy: 'metadata.score', order: 'desc', limit: k });
    return all.map(r => r.id);
  }
}
```

### Integration 3: Cross-Repo Knowledge Transfer

With ruvvector as the archive, **successful mutations become searchable across
repos**. A new `knowledge-transfer` mode:

```bash
# Publish this repo's winners to the shared fleet archive
metaharness-darwin publish --archive ruvvector --fleet-url http://ruvvector-fleet:6333

# Seed this repo's evolution with winners from similar repos
metaharness-darwin seed --fleet-url http://ruvvector-fleet:6333 --similarity 0.8
```

Similarity is computed on the `RepoProfile` embedding (language, framework,
task distribution). A Python/Go agent repo seeded with winning `retry_policy.ts`
mutations from other Python/Go repos converges faster in early generations.

### Integration 4: RDT Depth Signal as Mutation Difficulty Router

The RDT model (ADR-258) records per-token halt depth via `DepthTelemetry`. For
mutation tasks:
- **Low halt depth** (≤ 3 loops) → simple whitespace/rename mutations; ruvllm
  handles these efficiently with greedy decode
- **High halt depth** (≥ 6 loops) → complex restructuring mutations; ruvllm
  benefits from more recurrent iterations (higher max_loops in request)

The `RuvllmMutator` can expose an adaptive mode:

```typescript
// In RuvllmMutator.generateMutation():
const complexity = estimateSurfaceComplexity(input.parentCode);
const maxLoops = complexity > 0.7 ? 12 : 6;   // passed as x-ruvllm-max-loops header
```

Darwin Mode's mutation budget (currently fixed per generation) becomes
**compute-proportional** — harder mutations get deeper inference.

---

## Deep Review: Is Darwin Mode Useful in MetaHarness?

### Yes — Five Concrete Reasons

**1. Closes the generation-to-improvement loop.**  
MetaHarness generates a harness. Darwin Mode improves it. Without Darwin Mode,
the harness is a static artifact. With Darwin Mode, it becomes a living system
that self-corrects on failure patterns.

**2. The frozen-kernel scorer maps to ruvvector's domain.**  
The Darwin Mode scorer's 6 base terms (taskSuccess, testPassRate, traceQuality,
costEfficiency, latencyEfficiency, safetyScore) are numerically well-defined.
These can be stored as structured metadata in ruvvector and used as secondary
sort keys in retrieval — exactly how ruvvector supports multi-objective queries.

**3. The 7-surface constraint is a natural HNSW namespace.**  
Each of the 7 mutation surfaces evolves independently. A per-surface HNSW index
in ruvvector lets the system ask "what are the 10 best `retry_policy.ts` variants
globally?" without conflating surfaces. ruvvector's namespace API maps directly.

**4. Darwin Mode's archive is a vector search problem in disguise.**  
The MAP-Elites and behavioral-diversity strategies already compute cosine distances
on 384-dim embeddings. This is HNSW's native workload. The filesystem archive is
an impedance mismatch; ruvvector removes it.

**5. ruvllm's GPU optimizations make local evolution viable.**  
Before ADR-258, a local 7B code model at Q4 took ~2 s per mutation on CPU.
After ADR-258 (KV pre-alloc, vectorized ACT, CUDA 13 support), a local
RDT/OpenMythos model takes ~300 ms on an RTX 5080. 4 children × 5 generations
= 20 mutations × 300 ms = 6 s total inference time for a full sweep. This makes
the local path **faster than the OpenRouter path** (4 × 5 × 500 ms = 10 s minimum
at typical API latency).

### What Darwin Mode Does Not Do (Boundaries)

Darwin Mode is **NOT**:
- A training loop (no gradient descent, no weight updates)
- A test-time compute primitive (inference depth is not modulated per token)
- A replacement for the base LLM (the frozen model is still the core reasoner)
- An autonomous system (it requires a human to define the task suite)

These boundaries are the right ones. Darwin Mode's thesis —  "frozen model,
evolving harness" — is orthogonal to ruvllm's thesis — "GPU-resident inference
for recurrent depth models." They compose without conflict.

---

## Consequences

### Immediate (with ADR-259 already in place)

| Capability | Before | After |
|-----------|--------|-------|
| Local evolution | No | Yes (--mutator ruvllm) |
| API cost per sweep | $0.15–$0.30 | $0 |
| Latency per mutation | 200–800 ms | 50–300 ms |

### With this ADR (ruvvector archive)

| Capability | Before | After |
|-----------|--------|-------|
| Selection speed (100 variants) | O(n) scan | O(log n) ANN |
| Cross-repo transfer | Impossible | Fleet archive via ruvvector |
| Surface-level search | No | Per-namespace HNSW index |
| Behavioral diversity selection | O(n²) pairwise | O(k log n) ANN |

### With Integration 4 (RDT depth router)

| Capability | Before | After |
|-----------|--------|-------|
| Mutation compute budget | Fixed | Complexity-proportional |
| Easy mutation latency | Same as hard | 2–3× faster |
| Hard mutation quality | Same as easy | Deeper reasoning (more loops) |

---

## Implementation Plan

| Step | Owner | Effort |
|------|-------|--------|
| 1. `ruvllm-mutator.ts` (ADR-259) | darwin-mode | ~80 LOC |
| 2. `archive-ruvvector.ts` | darwin-mode | ~120 LOC |
| 3. CLI flags `--archive ruvvector --db-path` | darwin-mode | ~20 LOC |
| 4. Per-surface HNSW namespaces | ruvvector (existing) | config only |
| 5. `publish` / `seed` fleet commands | darwin-mode | ~200 LOC |
| 6. RDT depth signal in `RuvllmMutator` | darwin-mode / ruvllm-cli | ~30 LOC |

Steps 1–3 are independent of ruvvector changes (ruvvector API is stable).
Steps 4–6 can ship in any order after Steps 1–3.

---

## Acceptance Test

The integration is complete when this pipeline passes end-to-end:

```bash
# Terminal 1: start ruvllm server (ADR-259)
ruvllm serve --model ~/.cache/models/deepseek-coder-7b-q4.gguf --port 8080

# Terminal 2: run evolution with ruvvector archive
metaharness-darwin evolve /path/to/test-agent-repo \
  --mutator ruvllm \
  --archive ruvvector \
  --db-path .ruvvector/darwin.db \
  --generations 5 \
  --children 4 \
  --selection quality-diversity

# Expected outcome:
# - Generation 5 best variant scores > generation 1 best variant by ≥ 5%
# - ruvvector archive contains 20 indexed variants
# - No OpenRouter API calls in network log
# - Total elapsed < 120 s on RTX 5080
```

---

## Alternatives Considered

**A. Chromadb / Qdrant as archive backend.**  
External vector databases require a server process and network hop. ruvvector is
embedded (same process, shared memory), which matches darwin-mode's lightweight
constraint. Additionally, ruvvector is already a first-party dependency across
the ruvector ecosystem.

**B. SQLite with cosine extension.**  
No ANN — O(n) scan even with the cosine extension. Suitable for < 50 variants
but degrades quadratically at fleet scale.

**C. Keep the filesystem archive, add a vector index as a sidecar.**  
Dual-write increases complexity and introduces consistency hazards. Replacing the
archive backend is cleaner and eliminates the sidecar.
