/**
 * SIFT-1M HNSW benchmark using hnswlib-node for comparison against ruvector.
 *
 * Usage:
 *   node scripts/sift1m_hnswlib_bench.mjs [data_dir] [corpus_limit] [query_limit]
 *
 * Reads from fvecs/ivecs files directly (no HDF5 needed).
 * Sweeps ef_search at [20, 40, 80, 100, 200, 400] to produce recall@10 vs QPS.
 */

import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { HierarchicalNSW } = require('hnswlib-node');

// ---------------------------------------------------------------------------
// fvecs / ivecs readers
// ---------------------------------------------------------------------------

function readFvecs(filePath, maxVecs = Infinity) {
  const fd = fs.openSync(filePath, 'r');
  const vecs = [];
  let dims = 0;
  const dimBuf = Buffer.allocUnsafe(4);

  while (vecs.length < maxVecs) {
    const bytesRead = fs.readSync(fd, dimBuf, 0, 4, null);
    if (bytesRead < 4) break;
    const d = dimBuf.readUInt32LE(0);
    if (dims === 0) dims = d;
    else if (d !== dims) throw new Error(`Inconsistent dims: expected ${dims}, got ${d}`);

    const vecBuf = Buffer.allocUnsafe(d * 4);
    fs.readSync(fd, vecBuf, 0, d * 4, null);
    const vec = new Array(d);
    for (let i = 0; i < d; i++) vec[i] = vecBuf.readFloatLE(i * 4);
    vecs.push(vec);
  }
  fs.closeSync(fd);
  return { vecs, dims };
}

function readIvecs(filePath, maxVecs = Infinity) {
  const fd = fs.openSync(filePath, 'r');
  const vecs = [];
  let dims = 0;
  const dimBuf = Buffer.allocUnsafe(4);

  while (vecs.length < maxVecs) {
    const bytesRead = fs.readSync(fd, dimBuf, 0, 4, null);
    if (bytesRead < 4) break;
    const d = dimBuf.readUInt32LE(0);
    if (dims === 0) dims = d;
    else if (d !== dims) throw new Error(`Inconsistent dims in ivecs`);

    const vecBuf = Buffer.allocUnsafe(d * 4);
    fs.readSync(fd, vecBuf, 0, d * 4, null);
    const vec = new Int32Array(d);
    for (let i = 0; i < d; i++) vec[i] = vecBuf.readInt32LE(i * 4);
    vecs.push(vec);
  }
  fs.closeSync(fd);
  return { vecs, dims };
}

// ---------------------------------------------------------------------------
// Recall@k
// ---------------------------------------------------------------------------

function recallAtK(resultIds, groundTruth, k) {
  const gt = new Set(Array.from(groundTruth).slice(0, k));
  let hits = 0;
  const limit = Math.min(resultIds.length, k);
  for (let i = 0; i < limit; i++) {
    if (gt.has(resultIds[i])) hits++;
  }
  return hits / Math.min(k, gt.size);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const dataDir = args[0] || 'bench_data/sift';
const corpusLimit = parseInt(args[1] || '1000000', 10);
const queryLimit = parseInt(args[2] || '10000', 10);
const M = 16;
const efConstruction = 200;
const k = 10;
const efValues = [20, 40, 80, 100, 200, 400];

console.log('=== SIFT-1M HNSW Benchmark (hnswlib-node) ===');
console.log(`Data dir   : ${dataDir}`);
console.log(`Corpus cap : ${corpusLimit}`);
console.log(`Query  cap : ${queryLimit}`);
console.log(`M          : ${M}`);
console.log(`efConstruct: ${efConstruction}`);
console.log(`k          : ${k}`);
console.log(`ef sweep   : ${efValues.join(',')}`);
console.log();

// Load data
process.stdout.write('Loading corpus... ');
let t = Date.now();
const { vecs: corpus, dims } = readFvecs(path.join(dataDir, 'sift_base.fvecs'), corpusLimit);
console.log(`${corpus.length} vectors × ${dims}d  (${((Date.now()-t)/1000).toFixed(1)}s)`);

process.stdout.write('Loading queries... ');
t = Date.now();
const { vecs: queries } = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queryLimit);
console.log(`${queries.length} vectors × ${dims}d  (${((Date.now()-t)/1000).toFixed(1)}s)`);

process.stdout.write('Loading ground truth... ');
t = Date.now();
const { vecs: groundTruth } = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queryLimit);
console.log(`${groundTruth.length} lists  (${((Date.now()-t)/1000).toFixed(1)}s)`);
console.log();

// Build index
console.log(`Building HNSW index (M=${M}, efC=${efConstruction})...`);
const hnsw = new HierarchicalNSW('l2', dims);
hnsw.initIndex(corpusLimit, M, efConstruction, 0 /* seed */);

const tBuild = Date.now();
for (let i = 0; i < corpus.length; i++) {
  hnsw.addPoint(corpus[i], i);
  if (i > 0 && i % 100_000 === 0) {
    const elapsed = (Date.now() - tBuild) / 1000;
    const rate = i / elapsed;
    console.log(`  Inserted ${i} (${rate.toFixed(0)} vec/s, ${elapsed.toFixed(1)}s elapsed)`);
  }
}
const buildSecs = (Date.now() - tBuild) / 1000;
const buildRate = corpus.length / buildSecs;
const memMB = (corpus.length * dims * 4) / (1024 * 1024) * 1.5;
console.log(`Build done: ${buildSecs.toFixed(1)}s  (${buildRate.toFixed(0)} vec/s, ${memMB.toFixed(0)} MB estimated)`);
console.log();

// ef_search sweep
const header = 'ef'.padEnd(8) + '  ' + 'recall@10'.padStart(10) + '  ' + 'QPS'.padStart(10) + '  ' + 'p50_us'.padStart(12) + '  ' + 'p99_us'.padStart(10);
console.log(header);
console.log('-'.repeat(58));

for (const ef of efValues) {
  hnsw.setEf(ef);
  const latenciesNs = [];
  let recallSum = 0;

  for (let qi = 0; qi < queries.length; qi++) {
    const tq = process.hrtime.bigint();
    const result = hnsw.searchKnn(queries[qi], k);
    const elapsedNs = Number(process.hrtime.bigint() - tq);
    latenciesNs.push(elapsedNs);

    recallSum += recallAtK(result.neighbors, groundTruth[qi], k);
  }

  const meanRecall = recallSum / queries.length;
  const totalS = latenciesNs.reduce((a, b) => a + b, 0) / 1e9;
  const qps = queries.length / totalS;

  const sorted = [...latenciesNs].sort((a, b) => a - b);
  const p50Us = sorted[Math.floor(sorted.length * 0.50)] / 1000;
  const p99Us = sorted[Math.floor(sorted.length * 0.99)] / 1000;

  console.log(
    String(ef).padEnd(8) + '  ' +
    meanRecall.toFixed(4).padStart(10) + '  ' +
    qps.toFixed(0).padStart(10) + '  ' +
    p50Us.toFixed(1).padStart(12) + '  ' +
    p99Us.toFixed(1).padStart(10)
  );
}

console.log();
console.log('Done.');
