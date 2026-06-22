# Harness optimization with Darwin Mode

MetaBioHacker optimizes its reconstruction pipeline with
[`@metaharness/darwin`](https://www.npmjs.com/package/@metaharness/darwin) using
the **"freeze the model, evolve the harness"** principle.

- **Frozen model** — the Rust acoustic engine (`sonic_ct` → WASM). The physics
  kernel is never mutated.
- **Evolved harness** — a reconstruction genome covering *what* is
  reconstructed, *how it is routed* (cheap → frontier), and *how it is scored*:

  ```
  reconstruction: voxelResolutionMm, temporalWindowMs, smoothing,
                  organPrior, confidenceThreshold, elements, fan
  modelRouting:   firstPass(local), escalation(cheap|mid|frontier),
                  frontierOnlyWhen { lowConfidence, inconsistentFrames, ... }
  scoring:        weights for shape / residual / latency / cost / safety
  ```

## Architecture

`evolve()` in `@metaharness/darwin` is its *code-surface* evolver (it mutates
harness source files against a task sandbox — built for LLM agent harnesses).
For MetaBioHacker's numeric genome we keep the same invariant —
**genome → run frozen engine → scored candidate → Pareto frontier** — using
Darwin's `mapLimit` (bounded-concurrency evaluation) and `paretoFront`
(multi-objective selection) primitives, plus an **archive** of every evaluated
variant.

### Cheap → frontier write layer (OpenRouter)

The model-routing tier is real. The local Rust reconstruction always runs first.
An **OpenRouter** LLM acts as the *write layer* that proposes harness genome
mutations (the Meta Harness idea: optimize the code around the model, not the
weights). Routing is honest and bounded:

- The cheaper model (`openai/gpt-4o-mini`) proposes most mutations.
- The **frontier** model (`openai/gpt-4o`) fires **only** when low-confidence
  slices are detected (`frontierCalls > 0`) and the genome's escalation policy
  allows it.
- LLM output is parsed, validated, and clamped before it can be evaluated, and
  it **never overrides physics** — it only proposes a policy mutation that the
  frozen Rust engine then re-scores.
- Hard cap of `LLM_BUDGET = 10` calls/run keeps spend trivial (~$0.001). With no
  `OPENROUTER_API_KEY` it falls back to the deterministic random mutator.

## Multi-objective fitness

`paretoFront` maximises shape score, temporal stability, and safety while
minimising acoustic residual, latency, and cost. Each is measured against the
frozen engine: shape = mean Dice; stability = `1 − stddev(per-slice Dice)`;
residual = mean speed MAE / window; latency/cost include simulated frontier
model-call economics; safety is penalised by high-severity quality flags.

## Acceptance test

A run passes when a Pareto-superior, **gate-passing** variant exists in the
archive: it improves temporal stability by ≥10% **or** latency by ≥20% with **no
regression** in acoustic residual, safety, or frontier model-calls.

```
candidates evaluated: 24 | gate-passing: 17
accepted: stability gain 2.7% | latency gain 92.8% | no-regress true
PASS — Pareto-superior harness found (freeze model, evolve harness)
LLM frontier-mutator calls: 10
```

The big win is compute arbitrage: the evolved harness reaches comparable shape
fidelity while routing far fewer slices to the expensive frontier tier.

```bash
cd examples/sonic-ct
export OPENROUTER_API_KEY=...   # optional; omit to use the random mutator
npm run optimize                # writes optimize.report.json
node probeDarwin.mjs            # verify the @metaharness/darwin export surface
```

The key is read from the environment only — never committed.

## Typed evolution module + tests

`src/optimizer/reconstructionEvolution.ts` is the faithful, typed implementation
of the invariant: a `MetaBioGenome` (reconstruction / routing / scoring / safety),
`runFrozenRustEngine` (spawns the real `sonic_ct_serve` binary over JSON stdio),
`shouldUseFrontier` / `routeReconstruction` (cheap → frontier, augmenting the
engine result without ever rewriting anatomy), `scoreCandidate`, `mutateGenome`,
and `evolveMetaBioHarness` (Darwin `mapLimit` + `paretoFront` + archive). The
frozen physics layer is the Rust binary `sonic_ct_serve` (input/output JSON).

`evolve()` from `@metaharness/darwin` is reserved for *actual source-surface*
mutation later; for now MetaBioHacker evolves the numeric harness genome.

Tests (`npm test`, Node type-stripping): mapLimit bounds concurrency;
paretoFront keeps a slower-accurate and a faster-cheaper candidate while
dropping a dominated one; frontier routing never bypasses the frozen engine.
