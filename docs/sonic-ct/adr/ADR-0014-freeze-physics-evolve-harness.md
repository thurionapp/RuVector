# ADR-0014: Freeze the Physics Engine, Evolve the Reconstruction Harness

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The acoustic physics in `sonic_ct` (forward model, backprojection, SART, volume
reconstruction) is the part we trust least to "improve" by hand-tuning and least
want an LLM rewriting. It is deterministic, testable, and the source of truth for
what the data actually says. At the same time, almost everything that makes a
reconstruction *useful* — voxel resolution, temporal window, smoothing, organ
priors, confidence thresholds, model routing, scoring weights, and explanation
policy — is policy, not physics, and benefits from search.

[@metaharness/darwin](https://www.npmjs.com/package/@metaharness/darwin) frames
this cleanly: **"freeze the model, evolve the harness."** We adopt that split.

## Decision

We freeze the Rust acoustic engine and evolve only the reconstruction **harness**.

- The frozen engine is exposed as a process boundary: `sonic_ct_serve`
  (`crates/sonic-ct/src/bin/serve.rs`) reads one JSON object on stdin (sample +
  harness policy + safety) and writes one JSON object of scores on stdout. The
  comment is explicit: "The physics is frozen; the harness only changes how
  reconstruction is driven, never the engine." It maps policy fields
  (`voxelResolutionMm`, `temporalWindowMs`, `smoothingAlpha`, prior weights) onto
  engine parameters but never alters the physics.
- The harness is a numeric genome, `MetaBioGenome`
  (`examples/sonic-ct/src/optimizer/reconstructionEvolution.ts`), carrying
  `reconstruction`, `routing`, `scoring`, and `safety` blocks. `engine.frozen` is
  a literal `true`.
- Evolution runs `genome -> runFrozenRustEngine -> scoreCandidate -> Pareto
  front -> mutate`, using Darwin's `mapLimit` (bounded-concurrency evaluation)
  and `paretoFront` (multi-objective selection) in `evolveMetaBioHarness`. Model
  routing escalates cheap → frontier only on low confidence or high disagreement
  (`routeReconstruction`), and a frontier model proposes a *policy* mutation — it
  never overrides anatomy.
- An optional LLM "write layer" (`examples/sonic-ct/optimize.mjs`) proposes
  harness mutations via OpenRouter under a hard call budget; its key is read from
  the environment only, and it falls back to the deterministic mutator when
  absent.

Acceptance gate: a variant is accepted only if it improves temporal stability by
≥10% **or** latency by ≥20%, with no regression in acoustic residual, safety, or
frontier model-calls (`isUsefulImprovement`; the `gate` in `optimize.mjs`).

## Consequences

### Positive

- The physics stays deterministic, auditable, and out of the mutation loop.
- Harness search is cheap, bounded, and reproducible (seeded RNG, capped LLM
  budget), and improvements are gated against hard non-regression criteria.

### Negative / Trade-offs

- Genuine gains that require physics changes are out of scope for this loop and
  must go through normal engineering review of the crate.
- The JSON-over-stdio boundary adds process overhead per evaluation.

## Alternatives Considered

- **Let the optimizer (or an LLM) tune the physics directly**: rejected — it
  removes the trusted, deterministic baseline and makes regressions hard to
  attribute.
- **Hand-tune harness parameters only**: rejected — slow and biased; the search
  loop explores the Pareto front far more thoroughly under explicit constraints.

## References

- `crates/sonic-ct/src/bin/serve.rs` (`sonic_ct_serve` — frozen engine over JSON stdio)
- `examples/sonic-ct/src/optimizer/reconstructionEvolution.ts` (`MetaBioGenome`, `runFrozenRustEngine`, `routeReconstruction`, `evolveMetaBioHarness`, `isUsefulImprovement`)
- `examples/sonic-ct/optimize.mjs` (OpenRouter write layer, bounded budget, env-only key, acceptance gate)
- [@metaharness/darwin — "freeze the model, evolve the harness"](https://www.npmjs.com/package/@metaharness/darwin)
