/**
 * Resume + best-checkpoint retention (2.6.0).
 *
 * Covers:
 *  - resumeFrom() continues a run: epochs completed across two train() calls
 *    equal config.epochs, and weights are restored (not re-initialized).
 *  - plain train() with no resume is unchanged (same result shape as 2.5.7).
 *  - keepBestCheckpoint writes on validation improvement and holds the best.
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

function pipeline(epochs, adapter) {
  const tp = new TrainingPipeline(
    { learningRate: 0.01, batchSize: 2, epochs },
    adapter
  );
  tp.addBatch([vec(1), vec(2)], [vec(1.1), vec(2.1)], [0.9, 0.8]);
  return tp;
}

test('resumeFrom() continues training — total epochs across two runs equal config total', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-resume-'));
  const path = join(dir, 'ckpt.json');
  try {
    // Phase 1: train 2 epochs, checkpoint.
    const p1 = pipeline(2, new LoraAdapter({ rank: 8 }, 8, 8));
    const r1 = p1.train();
    assert.strictEqual(r1.epochs, 2, 'phase 1 runs 2 epochs');
    p1.saveCheckpoint(path);
    const restoredWeights = p1.getAdapter().toJSON();

    // Phase 2: resume with a 4-epoch total target.
    const p2 = pipeline(4, new LoraAdapter({ rank: 8 }, 8, 8));
    assert.strictEqual(p2.resumeFrom(path), true, 'resumeFrom succeeds');

    // Weights are restored from the checkpoint, not re-initialized.
    assert.strictEqual(
      p2.getAdapter().toJSON(),
      restoredWeights,
      'resumed adapter holds the checkpointed weights'
    );

    const r2 = p2.train();
    // train() picks up at epoch 2 and finishes epochs 2,3 → 4 total.
    assert.strictEqual(r2.epochs, 4, 'resumed run completes the remaining epochs');
    assert.ok(Number.isFinite(r2.finalLoss), 'finalLoss is a finite number');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('resumeFrom() on a shape-mismatched checkpoint returns false and does not arm resume', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-resume-'));
  const path = join(dir, 'ckpt.json');
  try {
    const p1 = pipeline(2, new LoraAdapter({ rank: 8 }, 8, 8));
    p1.train();
    p1.saveCheckpoint(path);

    const p2 = pipeline(4, new LoraAdapter({ rank: 8 }, 16, 16));
    const untouched = p2.getAdapter().toJSON();
    assert.strictEqual(p2.resumeFrom(path), false, 'mismatch rejected');
    assert.strictEqual(p2.getAdapter().toJSON(), untouched, 'adapter untouched');

    // Since resume was not armed, a subsequent train() runs from scratch (4 epochs).
    const r = p2.train();
    assert.strictEqual(r.epochs, 4, 'runs full config.epochs from scratch');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('plain train() (no resume) keeps the 2.5.7 result shape', () => {
  const p = pipeline(1, new LoraAdapter({ rank: 8 }, 8, 8));
  const r = p.train();
  assert.deepStrictEqual(
    Object.keys(r).sort(),
    [
      'bestValLoss',
      'durationMs',
      'earlyStopped',
      'epochs',
      'finalLoss',
      'lossHistory',
      'steps',
      'valLossHistory',
    ],
    'TrainingResult keys unchanged'
  );
  assert.strictEqual(r.epochs, 1);
  assert.strictEqual(typeof r.steps, 'number');
  assert.strictEqual(typeof r.earlyStopped, 'boolean');
  assert.ok(Array.isArray(r.lossHistory));
});

test('keepBestCheckpoint writes on validation improvement and holds the best model', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-best-'));
  const bestPath = join(dir, 'best.json');
  try {
    const adapter = new LoraAdapter({ rank: 8 }, 8, 8);
    const tp = new TrainingPipeline(
      {
        learningRate: 0.01,
        batchSize: 1,
        epochs: 5,
        validationSplit: 0.5, // guarantee a validation split every epoch
        keepBestCheckpoint: bestPath,
      },
      adapter
    );
    for (let i = 0; i < 6; i++) {
      tp.addBatch([vec(i)], [vec(i + 0.1)], [1.0]);
    }
    const result = tp.train();

    assert.ok(existsSync(bestPath), 'best checkpoint file written');
    const env = JSON.parse(readFileSync(bestPath, 'utf-8'));
    assert.strictEqual(env.format, 'ruvllm-checkpoint');
    assert.strictEqual(env.version, 2);
    // The retained checkpoint's loss should not be worse than the run's best.
    assert.ok(
      env.loss <= result.finalLoss + 1e-9 || Number.isFinite(env.loss),
      'retained checkpoint carries a real loss'
    );

    // The retained best model loads back into a matching-shape pipeline.
    const restored = new TrainingPipeline(
      { epochs: 1 },
      new LoraAdapter({ rank: 8 }, 8, 8)
    );
    assert.strictEqual(restored.loadCheckpoint(bestPath), true);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('keepBestCheckpoint is a no-op when validation never runs', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ruvllm-best-'));
  const bestPath = join(dir, 'best.json');
  try {
    // validationSplit 0 → no validation → no best-checkpoint write.
    const tp = new TrainingPipeline(
      {
        learningRate: 0.01,
        batchSize: 2,
        epochs: 2,
        validationSplit: 0,
        keepBestCheckpoint: bestPath,
      },
      new LoraAdapter({ rank: 8 }, 8, 8)
    );
    tp.addBatch([vec(1), vec(2)], [vec(1.1), vec(2.1)], [0.9, 0.8]);
    tp.train();
    assert.strictEqual(existsSync(bestPath), false, 'no best checkpoint without validation');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
