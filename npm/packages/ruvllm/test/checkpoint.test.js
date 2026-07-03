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

const { TrainingPipeline } = require('../dist/cjs/index.js');

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
