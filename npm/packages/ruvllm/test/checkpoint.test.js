/**
 * Checkpoint persistence — regression for ruvnet/ruflo#2549 (downstream
 * report): saveCheckpoint() was private, ignored its argument, returned
 * undefined, and wrote zero bytes. It must now persist to disk when given
 * a path and round-trip through loadCheckpoint(path).
 */
const { test } = require('node:test');
const assert = require('node:assert');
const { existsSync, readFileSync, rmSync, mkdtempSync } = require('node:fs');
const { join } = require('node:path');
const { tmpdir } = require('node:os');

const { TrainingPipeline, LoraAdapter } = require('../dist/cjs/index.js');

function vec(seed) {
  return Array.from({ length: 8 }, (_, i) => Math.sin(seed + i));
}

function trainedPipeline() {
  const tp = new TrainingPipeline({
    learningRate: 0.01,
    batchSize: 2,
    epochs: 1,
    inputDim: 8,
    outputDim: 8,
  });
  tp.addBatch([vec(1), vec(2)], [vec(1.1), vec(2.1)], [0.9, 0.8]);
  tp.train();
  return tp;
}

// Pipeline over an adapter of an explicit shape. The pipeline's own config
// does not drive adapter geometry, so shape-sensitive tests pass the adapter.
function shapedPipeline(inputDim, outputDim, rank = 8) {
  const adapter = new LoraAdapter({ rank }, inputDim, outputDim);
  const tp = new TrainingPipeline(
    { learningRate: 0.01, batchSize: 2, epochs: 1 },
    adapter
  );
  return tp;
}

function trainedShapedPipeline(inputDim, outputDim, rank = 8) {
  const tp = shapedPipeline(inputDim, outputDim, rank);
  tp.addBatch([vec(1), vec(2)], [vec(1.1), vec(2.1)], [0.9, 0.8]);
  tp.train();
  return tp;
}

test('saveCheckpoint() with no path records in-memory and returns metadata', () => {
  const tp = trainedPipeline();
  const r = tp.saveCheckpoint();
  assert.ok(r && typeof r === 'object', 'returns a result object, not undefined');
  assert.strictEqual(typeof r.index, 'number');
  assert.strictEqual(typeof r.loss, 'number');
  assert.strictEqual(r.path, undefined);
});

test('saveCheckpoint(path) writes a non-empty file and reports bytes', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'nested', 'ckpt.json');
  try {
    const tp = trainedPipeline();
    const r = tp.saveCheckpoint(path);
    assert.strictEqual(r.path, path);
    assert.ok(r.bytes > 0, `bytes must be > 0, got ${r.bytes}`);
    assert.ok(existsSync(path), 'file must exist on disk');
    const onDisk = JSON.parse(readFileSync(path, 'utf-8'));
    assert.strictEqual(onDisk.format, 'ruvllm-checkpoint');
    assert.strictEqual(typeof onDisk.weights, 'string');
    assert.strictEqual(readFileSync(path, 'utf-8').length, r.bytes);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('loadCheckpoint(path) round-trips weights from disk', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'ckpt.json');
  try {
    const tp = trainedPipeline();
    const before = tp.getAdapter().toJSON();
    tp.saveCheckpoint(path);

    // Fresh pipeline, load from disk.
    const tp2 = new TrainingPipeline({
      learningRate: 0.01,
      batchSize: 2,
      epochs: 1,
      inputDim: 8,
      outputDim: 8,
    });
    assert.strictEqual(tp2.loadCheckpoint(path), true);
    assert.strictEqual(tp2.getAdapter().toJSON(), before, 'weights round-trip');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('loadCheckpoint by in-memory index still works (back-compat)', () => {
  const tp = trainedPipeline();
  const r = tp.saveCheckpoint();
  assert.strictEqual(tp.loadCheckpoint(r.index), true);
});

test('loadCheckpoint rejects missing files and foreign JSON', () => {
  const tp = trainedPipeline();
  assert.strictEqual(tp.loadCheckpoint('/nonexistent/nope.json'), false);
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'foreign.json');
  try {
    require('node:fs').writeFileSync(path, JSON.stringify({ hello: 'world' }));
    assert.strictEqual(tp.loadCheckpoint(path), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---- v2 checkpoint metadata (2.6.0) ----

test('saveCheckpoint(path) writes v2 envelope with adapter geometry + pipeline config', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'ckpt.json');
  try {
    const tp = trainedShapedPipeline(8, 8, 8);
    tp.saveCheckpoint(path);
    const env = JSON.parse(readFileSync(path, 'utf-8'));
    assert.strictEqual(env.version, 2, 'envelope version bumped to 2');
    assert.deepStrictEqual(
      env.config,
      { inputDim: 8, outputDim: 8, rank: 8 },
      'config carries adapter geometry'
    );
    assert.strictEqual(env.pipelineConfig.learningRate, 0.01);
    assert.strictEqual(env.pipelineConfig.batchSize, 2);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('loadCheckpoint round-trips a v2 checkpoint into a matching-shape pipeline', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'ckpt.json');
  try {
    const tp = trainedShapedPipeline(8, 8, 8);
    const before = tp.getAdapter().toJSON();
    tp.saveCheckpoint(path);

    const tp2 = shapedPipeline(8, 8, 8);
    assert.strictEqual(tp2.loadCheckpoint(path), true);
    assert.strictEqual(tp2.getAdapter().toJSON(), before, 'weights round-trip');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('loadCheckpoint rejects a v2 checkpoint whose dims mismatch the adapter', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'ckpt.json');
  try {
    const tp = trainedShapedPipeline(8, 8, 8);
    tp.saveCheckpoint(path);
    const before = tp.getAdapter().toJSON();

    // Differently-shaped pipeline must refuse the mis-shaped weights...
    const mismatch = shapedPipeline(16, 16, 8);
    const untouched = mismatch.getAdapter().toJSON();
    assert.strictEqual(mismatch.loadCheckpoint(path), false, 'rejects dim mismatch');
    assert.strictEqual(
      mismatch.getAdapter().toJSON(),
      untouched,
      'adapter left untouched on rejection'
    );

    // ...and a rank mismatch is rejected too.
    const rankMismatch = shapedPipeline(8, 8, 4);
    assert.strictEqual(rankMismatch.loadCheckpoint(path), false, 'rejects rank mismatch');

    // Sanity: the matching shape still loads.
    const ok = shapedPipeline(8, 8, 8);
    assert.strictEqual(ok.loadCheckpoint(path), true);
    assert.strictEqual(ok.getAdapter().toJSON(), before);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('loadCheckpoint loads a v1 checkpoint regardless of dims (back-compat)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-ckpt-'));
  const path = join(dir, 'v1.json');
  try {
    // Hand-craft a v1 envelope (no config/pipelineConfig, version:1).
    const sourceAdapter = new LoraAdapter({ rank: 8 }, 8, 8);
    const v1 = {
      format: 'ruvllm-checkpoint',
      version: 1,
      epoch: 0,
      step: 1,
      loss: 0.5,
      weights: sourceAdapter.toJSON(),
      timestamp: Date.now(),
    };
    require('node:fs').writeFileSync(path, JSON.stringify(v1));

    // A pipeline with DIFFERENT adapter dims must still load a v1 file —
    // v1 carries no geometry, so no shape check applies.
    const tp = shapedPipeline(16, 16, 8);
    assert.strictEqual(tp.loadCheckpoint(path), true, 'v1 loads without dim check');
    assert.strictEqual(tp.getAdapter().toJSON(), sourceAdapter.toJSON());
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
