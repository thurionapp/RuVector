/**
 * @fileoverview Cross-provider query/passage prefix contract test (#662 follow-up)
 *
 * Reads the SAME fixture the Rust side asserts against
 * (`fixtures/lattice-embed/query-prefixes.json` at the repo root, consumed by
 * `crates/ruvector-core/src/embeddings.rs`'s
 * `lattice_native::tests::cross_provider_query_prefix_contract`), so this
 * TS/WASM `LatticeWasmEmbeddings` provider and the Rust `LatticeEmbedding`
 * provider cannot silently re-diverge on which prefix a model gets -- the
 * failure mode #662 was, and #663 fixed for `bge-small` on this side.
 *
 * No model weights or the optional `@khive-ai/lattice-embed-wasm` peer
 * package are needed: `applyLatticeWasmQueryPrefix` is a pure function over
 * the model's canonical name, exercised directly (same approach the existing
 * `embeddings.test.ts` #662 tests already use).
 *
 * @author ruv.io Team <info@ruv.io>
 * @license MIT
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { applyLatticeWasmQueryPrefix } from '../src/embeddings.js';

interface PrefixFixtureEntry {
  query_prefix: string | null;
  passage_prefix: string | null;
}

interface PrefixFixture {
  models: Record<string, PrefixFixtureEntry>;
}

const FIXTURE_PATH = fileURLToPath(
  new URL('../../../../fixtures/lattice-embed/query-prefixes.json', import.meta.url)
);

function loadFixture(): PrefixFixture {
  const raw = readFileSync(FIXTURE_PATH, 'utf8');
  return JSON.parse(raw) as PrefixFixture;
}

const fixture = loadFixture();
const modelNames = Object.keys(fixture.models);

describe('LatticeWasmEmbeddings cross-provider query prefix contract (#662 follow-up)', () => {
  it('the shared fixture is non-empty and covers bge-small + minilm', () => {
    assert.ok(modelNames.length > 0, 'fixture must not be empty');
    assert.ok(
      modelNames.includes('bge-small'),
      "fixture must cover 'bge-small' -- the model #662 was about"
    );
    assert.ok(
      modelNames.includes('minilm'),
      "fixture must cover 'minilm' as the symmetric control case"
    );
  });

  for (const model of modelNames) {
    const expected = fixture.models[model];

    it(`applyLatticeWasmQueryPrefix('${model}', ...) matches the shared fixture's query_prefix`, () => {
      const text = 'contract-test-probe';
      const expectedOutput = expected.query_prefix ? `${expected.query_prefix}${text}` : text;
      assert.strictEqual(
        applyLatticeWasmQueryPrefix(model, text),
        expectedOutput,
        `model '${model}': expected query prefix ${JSON.stringify(expected.query_prefix)} ` +
          `from fixtures/lattice-embed/query-prefixes.json to match ` +
          `LATTICE_WASM_QUERY_INSTRUCTIONS in src/embeddings.ts. If lattice-embed intentionally ` +
          `changed this model's convention, update the fixture AND the Rust sibling test in ` +
          `crates/ruvector-core/src/embeddings.rs (lattice_native::tests::` +
          `cross_provider_query_prefix_contract) together.`
      );
    });

    it(`'${model}' has no TS-side passage prefix, so the fixture's passage_prefix must be null`, () => {
      // LatticeWasmEmbeddings never prefixes the passage/document side --
      // embedText/embedTexts pass text through unchanged, and there is no
      // applyLatticeWasmPassagePrefix counterpart to call. If this assertion
      // ever fails, a model in the fixture has grown a non-null
      // passage_prefix and this provider needs a matching passage-prefix
      // path before the assertion (and this comment) can be updated
      // honestly.
      assert.strictEqual(
        expected.passage_prefix,
        null,
        `model '${model}': fixture declares a non-null passage_prefix but ` +
          `LatticeWasmEmbeddings has no passage-prefix implementation to check it against`
      );
    });
  }
});
