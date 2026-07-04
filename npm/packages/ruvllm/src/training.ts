/**
 * Training Pipeline for SONA
 *
 * Comprehensive training infrastructure with metrics tracking,
 * learning rate scheduling, and checkpoint management.
 *
 * @example
 * ```typescript
 * import { TrainingPipeline, TrainingConfig } from '@ruvector/ruvllm';
 *
 * const pipeline = new TrainingPipeline({
 *   learningRate: 0.001,
 *   batchSize: 32,
 *   epochs: 10,
 * });
 *
 * // Add training data
 * pipeline.addBatch(inputs, targets, qualities);
 *
 * // Run training
 * const result = pipeline.train();
 * console.log(`Final loss: ${result.finalLoss}`);
 * ```
 */

import { writeFileSync, readFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';
import { Embedding, TrainingConfig, TrainingResult } from './types';
import { LoraAdapter } from './lora';
import { EwcManager } from './sona';

/**
 * Default training config
 */
const DEFAULT_TRAINING_CONFIG: Required<TrainingConfig> = {
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  scheduler: 'cosine',
  warmupSteps: 100,
  weightDecay: 0.01,
  gradientClip: 1.0,
  earlyStoppingPatience: 3,
  checkpointInterval: 1,
  ewcLambda: 2000,
  validationSplit: 0.1,
  keepBestCheckpoint: '',
};

/**
 * Training metrics
 */
export interface TrainingMetrics {
  /** Current epoch */
  epoch: number;
  /** Current step */
  step: number;
  /** Training loss */
  trainLoss: number;
  /** Validation loss */
  valLoss: number;
  /** Learning rate */
  learningRate: number;
  /** Gradient norm */
  gradNorm: number;
  /** Steps per second */
  stepsPerSecond: number;
  /** ETA in seconds */
  etaSeconds: number;
}

/**
 * Training data batch
 */
export interface TrainingBatch {
  /** Input embeddings */
  inputs: Embedding[];
  /** Target outputs */
  targets: Embedding[];
  /** Quality scores */
  qualities: number[];
}

/**
 * Checkpoint data
 */
export interface Checkpoint {
  /** Epoch number */
  epoch: number;
  /** Step number */
  step: number;
  /** Training loss at checkpoint */
  loss: number;
  /** Model weights (serialized) */
  weights: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Result of a saveCheckpoint() call.
 */
export interface CheckpointSaveResult {
  /** Index of the checkpoint in the in-memory checkpoint list */
  index: number;
  /** Epoch the checkpoint captured */
  epoch: number;
  /** Step the checkpoint captured */
  step: number;
  /** Training loss at checkpoint time */
  loss: number;
  /** Absolute or as-given file path, when persisted to disk */
  path?: string;
  /** Serialized size in bytes, when persisted to disk */
  bytes?: number;
}

/**
 * On-disk checkpoint envelope version.
 *
 * v1 — {format, version:1, epoch, step, loss, weights, timestamp}. No adapter
 *      geometry, so loadCheckpoint() could not detect a shape mismatch.
 * v2 — adds {config:{inputDim, outputDim, rank}, pipelineConfig:{learningRate,
 *      batchSize}} so loadCheckpoint() can reject weights that don't fit the
 *      current adapter. v1 files still load (back-compat) with no shape check.
 */
const CHECKPOINT_FORMAT_VERSION = 2;

/**
 * Learning Rate Scheduler
 */
export class LRScheduler {
  private config: Required<TrainingConfig>;
  private initialLR: number;
  private currentStep: number = 0;
  private totalSteps: number;

  constructor(config: Required<TrainingConfig>, totalSteps: number) {
    this.config = config;
    this.initialLR = config.learningRate;
    this.totalSteps = totalSteps;
  }

  /**
   * Get learning rate for current step
   */
  getLR(): number {
    switch (this.config.scheduler) {
      case 'constant':
        return this.initialLR;

      case 'linear':
        return this.initialLR * (1 - this.currentStep / this.totalSteps);

      case 'cosine':
        return this.initialLR * 0.5 * (1 + Math.cos(Math.PI * this.currentStep / this.totalSteps));

      case 'warmup':
        if (this.currentStep < this.config.warmupSteps) {
          return this.initialLR * (this.currentStep / this.config.warmupSteps);
        }
        // Cosine decay after warmup
        const decaySteps = this.totalSteps - this.config.warmupSteps;
        const decayProgress = (this.currentStep - this.config.warmupSteps) / decaySteps;
        return this.initialLR * 0.5 * (1 + Math.cos(Math.PI * decayProgress));

      default:
        return this.initialLR;
    }
  }

  /**
   * Step the scheduler
   */
  step(): void {
    this.currentStep++;
  }

  /**
   * Reset scheduler
   */
  reset(): void {
    this.currentStep = 0;
  }
}

/**
 * Training Metrics Tracker
 */
export class MetricsTracker {
  private lossHistory: number[] = [];
  private valLossHistory: number[] = [];
  private gradNormHistory: number[] = [];
  private startTime: number = Date.now();
  private stepTimes: number[] = [];

  /**
   * Record training loss
   */
  recordLoss(loss: number): void {
    this.lossHistory.push(loss);
  }

  /**
   * Record validation loss
   */
  recordValLoss(loss: number): void {
    this.valLossHistory.push(loss);
  }

  /**
   * Record gradient norm
   */
  recordGradNorm(norm: number): void {
    this.gradNormHistory.push(norm);
  }

  /**
   * Record step time
   */
  recordStepTime(ms: number): void {
    this.stepTimes.push(ms);
  }

  /**
   * Get average loss over last N steps
   */
  avgLoss(n: number = 100): number {
    const recent = this.lossHistory.slice(-n);
    return recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  }

  /**
   * Get average validation loss
   */
  avgValLoss(n: number = 10): number {
    const recent = this.valLossHistory.slice(-n);
    return recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  }

  /**
   * Get steps per second
   */
  stepsPerSecond(): number {
    if (this.stepTimes.length === 0) return 0;
    const avgStepTime = this.stepTimes.slice(-100).reduce((a, b) => a + b, 0) / Math.min(this.stepTimes.length, 100);
    return avgStepTime > 0 ? 1000 / avgStepTime : 0;
  }

  /**
   * Get ETA in seconds
   */
  eta(remainingSteps: number): number {
    const sps = this.stepsPerSecond();
    return sps > 0 ? remainingSteps / sps : 0;
  }

  /**
   * Get best validation loss
   */
  bestValLoss(): number {
    return this.valLossHistory.length > 0 ? Math.min(...this.valLossHistory) : Infinity;
  }

  /**
   * Get total duration
   */
  duration(): number {
    return Date.now() - this.startTime;
  }

  /**
   * Get all loss history
   */
  getLossHistory(): number[] {
    return [...this.lossHistory];
  }

  /**
   * Get all validation loss history
   */
  getValLossHistory(): number[] {
    return [...this.valLossHistory];
  }

  /**
   * Reset tracker
   */
  reset(): void {
    this.lossHistory = [];
    this.valLossHistory = [];
    this.gradNormHistory = [];
    this.stepTimes = [];
    this.startTime = Date.now();
  }
}

/**
 * Training Pipeline
 *
 * Full training infrastructure for SONA models.
 */
export class TrainingPipeline {
  private config: Required<TrainingConfig>;
  private adapter: LoraAdapter;
  private ewcManager: EwcManager;
  private metrics: MetricsTracker;
  private scheduler: LRScheduler | null = null;
  private batches: TrainingBatch[] = [];
  private checkpoints: Checkpoint[] = [];
  private currentEpoch: number = 0;
  private currentStep: number = 0;
  private bestValLoss: number = Infinity;
  private patienceCounter: number = 0;
  /** Set by resumeFrom(); makes the next train() continue instead of restart. */
  private resumePending: boolean = false;

  constructor(config?: TrainingConfig, adapter?: LoraAdapter) {
    this.config = { ...DEFAULT_TRAINING_CONFIG, ...config };
    this.adapter = adapter || new LoraAdapter({ rank: 8 });
    this.ewcManager = new EwcManager(this.config.ewcLambda);
    this.metrics = new MetricsTracker();
  }

  /**
   * Add training batch
   */
  addBatch(inputs: Embedding[], targets: Embedding[], qualities: number[]): void {
    this.batches.push({ inputs, targets, qualities });
  }

  /**
   * Add training data
   */
  addData(data: Array<{ input: Embedding; target: Embedding; quality: number }>): void {
    // Group into batches
    for (let i = 0; i < data.length; i += this.config.batchSize) {
      const batch = data.slice(i, i + this.config.batchSize);
      this.addBatch(
        batch.map(d => d.input),
        batch.map(d => d.target),
        batch.map(d => d.quality)
      );
    }
  }

  /**
   * Run training.
   *
   * When {@link resumeFrom} primed the pipeline, this continues from the
   * restored epoch (running the remaining epochs of `config.epochs`) and
   * advances the LR scheduler to the restored step so the schedule is
   * unbroken; metrics history is preserved. Without a resume the run is
   * unchanged from a fresh start — same reset, same scheduler, same result
   * shape as before this method learned to resume.
   */
  train(): TrainingResult {
    const resuming = this.resumePending;
    this.resumePending = false;
    // currentEpoch is the last COMPLETED epoch index, so resume the next one.
    const startEpoch = resuming ? this.currentEpoch + 1 : 0;

    const totalSteps = this.batches.length * this.config.epochs;
    this.scheduler = new LRScheduler(this.config, totalSteps);
    if (resuming) {
      // Fast-forward the fresh scheduler to the restored step.
      for (let s = 0; s < this.currentStep && s < totalSteps; s++) {
        this.scheduler.step();
      }
    } else {
      this.metrics.reset();
    }
    this.adapter.startTraining(this.config.learningRate);

    let earlyStopped = false;

    for (let epoch = startEpoch; epoch < this.config.epochs; epoch++) {
      this.currentEpoch = epoch;

      // Shuffle batches
      const shuffledBatches = this.shuffleBatches();

      // Split into train/val
      const valSize = Math.floor(shuffledBatches.length * this.config.validationSplit);
      const trainBatches = shuffledBatches.slice(valSize);
      const valBatches = shuffledBatches.slice(0, valSize);

      // Training epoch
      for (const batch of trainBatches) {
        const stepStart = Date.now();
        const loss = this.trainStep(batch);
        this.metrics.recordLoss(loss);
        this.metrics.recordStepTime(Date.now() - stepStart);
        this.scheduler.step();
        this.currentStep++;
      }

      // Validation
      if (valBatches.length > 0) {
        const valLoss = this.validate(valBatches);
        this.metrics.recordValLoss(valLoss);

        // Early stopping
        if (valLoss < this.bestValLoss) {
          this.bestValLoss = valLoss;
          this.patienceCounter = 0;
          // Retain the best-validation model to a stable path when configured.
          if (this.config.keepBestCheckpoint) {
            this.saveCheckpoint(this.config.keepBestCheckpoint);
          }
        } else {
          this.patienceCounter++;
          if (this.patienceCounter >= this.config.earlyStoppingPatience) {
            earlyStopped = true;
            break;
          }
        }
      }

      // Checkpoint
      if ((epoch + 1) % this.config.checkpointInterval === 0) {
        this.saveCheckpoint();
      }
    }

    this.adapter.endTraining();

    // Register with EWC for continual learning
    const weights = this.adapter.merge().flat();
    this.ewcManager.registerTask(`task-${Date.now()}`, weights);

    return {
      epochs: this.currentEpoch + 1,
      steps: this.currentStep,
      finalLoss: this.metrics.avgLoss(100),
      bestValLoss: this.bestValLoss,
      durationMs: this.metrics.duration(),
      lossHistory: this.metrics.getLossHistory(),
      valLossHistory: this.metrics.getValLossHistory(),
      earlyStopped,
    };
  }

  /**
   * Single training step
   */
  private trainStep(batch: TrainingBatch): number {
    let totalLoss = 0;
    const lr = this.scheduler?.getLR() || this.config.learningRate;

    for (let i = 0; i < batch.inputs.length; i++) {
      const input = batch.inputs[i];
      const target = batch.targets[i];
      const quality = batch.qualities[i];

      // Forward pass
      const output = this.adapter.forward(input);

      // Compute loss (MSE weighted by quality)
      const gradOutput: number[] = [];
      let loss = 0;
      for (let j = 0; j < output.length; j++) {
        const diff = output[j] - (target[j] || 0);
        loss += diff * diff;
        gradOutput.push(2 * diff * quality); // Quality-weighted gradient
      }
      loss = (loss / output.length) * quality;

      // Add EWC penalty
      const ewcPenalty = this.ewcManager.computePenalty(this.adapter.merge().flat());
      loss += ewcPenalty * 0.001;

      // Backward pass
      this.adapter.backward(input, gradOutput, lr);

      totalLoss += loss;
    }

    return totalLoss / batch.inputs.length;
  }

  /**
   * Validation pass
   */
  private validate(batches: TrainingBatch[]): number {
    let totalLoss = 0;
    let count = 0;

    for (const batch of batches) {
      for (let i = 0; i < batch.inputs.length; i++) {
        const output = this.adapter.forward(batch.inputs[i]);
        const target = batch.targets[i];

        let loss = 0;
        for (let j = 0; j < output.length; j++) {
          const diff = output[j] - (target[j] || 0);
          loss += diff * diff;
        }
        totalLoss += loss / output.length;
        count++;
      }
    }

    return count > 0 ? totalLoss / count : 0;
  }

  /**
   * Save a checkpoint.
   *
   * Always records the checkpoint in the in-memory list (the behavior the
   * training loop relies on). When `path` is given, additionally persists
   * the checkpoint to disk as versioned JSON — before v2.5.7 this method
   * was private, ignored any argument, and never wrote a file, so callers
   * passing a path got `undefined` back and zero bytes on disk.
   *
   * @param path Optional file path; parent directories are created.
   * @returns Where the checkpoint went — in-memory index, and file
   *          path + byte size when persisted.
   */
  saveCheckpoint(path?: string): CheckpointSaveResult {
    const checkpoint: Checkpoint = {
      epoch: this.currentEpoch,
      step: this.currentStep,
      loss: this.metrics.avgLoss(100),
      weights: this.adapter.toJSON(),
      timestamp: Date.now(),
    };
    this.checkpoints.push(checkpoint);

    const result: CheckpointSaveResult = {
      index: this.checkpoints.length - 1,
      epoch: checkpoint.epoch,
      step: checkpoint.step,
      loss: checkpoint.loss,
    };

    if (path) {
      const envelope = {
        format: 'ruvllm-checkpoint',
        version: CHECKPOINT_FORMAT_VERSION,
        // Adapter geometry + pipeline hyperparams — lets loadCheckpoint()
        // reject weights that don't fit the current adapter (v2, see below).
        config: {
          inputDim: this.adapter.getInputDim(),
          outputDim: this.adapter.getOutputDim(),
          rank: this.adapter.getConfig().rank,
        },
        pipelineConfig: {
          learningRate: this.config.learningRate,
          batchSize: this.config.batchSize,
        },
        ...checkpoint,
      };
      const serialized = JSON.stringify(envelope);
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, serialized, 'utf-8');
      result.path = path;
      result.bytes = Buffer.byteLength(serialized, 'utf-8');
    }

    return result;
  }

  /**
   * Load a checkpoint — by in-memory index, or from a file previously
   * written by `saveCheckpoint(path)`.
   *
   * For v2 files, the envelope's adapter geometry is checked against the
   * current adapter first; a mismatch (different inputDim/outputDim/rank)
   * returns false and leaves the adapter untouched, so mis-shaped weights
   * are never silently restored. v1 files carry no geometry and load as
   * before (back-compat).
   */
  loadCheckpoint(indexOrPath: number | string): boolean {
    let checkpoint: Checkpoint | undefined;

    if (typeof indexOrPath === 'number') {
      checkpoint = this.checkpoints[indexOrPath];
    } else {
      try {
        const parsed = JSON.parse(readFileSync(indexOrPath, 'utf-8'));
        if (parsed?.format !== 'ruvllm-checkpoint' || typeof parsed.weights !== 'string') {
          return false;
        }
        if (typeof parsed.version === 'number' && parsed.version >= 2 && parsed.config) {
          const c = parsed.config;
          if (
            c.inputDim !== this.adapter.getInputDim() ||
            c.outputDim !== this.adapter.getOutputDim() ||
            c.rank !== this.adapter.getConfig().rank
          ) {
            return false;
          }
        }
        checkpoint = parsed as Checkpoint;
      } catch {
        return false;
      }
    }
    if (!checkpoint) return false;

    this.adapter = LoraAdapter.fromJSON(checkpoint.weights);
    this.currentEpoch = checkpoint.epoch;
    this.currentStep = checkpoint.step;
    return true;
  }

  /**
   * Resume training from a checkpoint file.
   *
   * Loads the checkpoint (with the same v2 shape validation as
   * {@link loadCheckpoint}) AND primes the pipeline so the next {@link train}
   * call continues from the restored epoch/step rather than restarting. This
   * is the explicit, least-invasive resume path: a plain `loadCheckpoint()`
   * still restores weights only (train() from scratch), while `resumeFrom()`
   * additionally makes the subsequent train() pick up where the run stopped.
   *
   * @returns true when the checkpoint loaded and resume was primed; false if
   *          the file was missing, foreign, or shape-mismatched (in which case
   *          no resume is armed).
   */
  resumeFrom(path: string): boolean {
    if (!this.loadCheckpoint(path)) return false;
    this.resumePending = true;
    return true;
  }

  /**
   * Get current metrics
   */
  getMetrics(): TrainingMetrics {
    return {
      epoch: this.currentEpoch,
      step: this.currentStep,
      trainLoss: this.metrics.avgLoss(100),
      valLoss: this.metrics.avgValLoss(10),
      learningRate: this.scheduler?.getLR() || this.config.learningRate,
      gradNorm: 0,
      stepsPerSecond: this.metrics.stepsPerSecond(),
      etaSeconds: this.metrics.eta(
        (this.config.epochs - this.currentEpoch) * this.batches.length
      ),
    };
  }

  /**
   * Get adapter
   */
  getAdapter(): LoraAdapter {
    return this.adapter;
  }

  /**
   * Get EWC manager
   */
  getEwcManager(): EwcManager {
    return this.ewcManager;
  }

  /**
   * Get checkpoints
   */
  getCheckpoints(): Checkpoint[] {
    return [...this.checkpoints];
  }

  /**
   * Reset pipeline
   */
  reset(): void {
    this.batches = [];
    this.checkpoints = [];
    this.currentEpoch = 0;
    this.currentStep = 0;
    this.bestValLoss = Infinity;
    this.patienceCounter = 0;
    this.resumePending = false;
    this.metrics.reset();
    this.adapter.reset();
  }

  private shuffleBatches(): TrainingBatch[] {
    const shuffled = [...this.batches];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
}

/**
 * Training Factory
 *
 * Create pre-configured training pipelines for common scenarios.
 */
export class TrainingFactory {
  /**
   * Create pipeline for quick fine-tuning
   */
  static quickFinetune(): TrainingPipeline {
    return new TrainingPipeline({
      learningRate: 0.01,
      epochs: 3,
      batchSize: 16,
      scheduler: 'constant',
    });
  }

  /**
   * Create pipeline for deep training
   */
  static deepTraining(): TrainingPipeline {
    return new TrainingPipeline({
      learningRate: 0.001,
      epochs: 50,
      batchSize: 32,
      scheduler: 'warmup',
      warmupSteps: 500,
      earlyStoppingPatience: 5,
    });
  }

  /**
   * Create pipeline for continual learning
   */
  static continualLearning(ewcLambda: number = 5000): TrainingPipeline {
    return new TrainingPipeline({
      learningRate: 0.0005,
      epochs: 10,
      batchSize: 16,
      scheduler: 'cosine',
      ewcLambda,
      earlyStoppingPatience: 10,
    });
  }

  /**
   * Create pipeline for federated aggregation
   */
  static federatedAggregation(): TrainingPipeline {
    return new TrainingPipeline({
      learningRate: 0.0001,
      epochs: 5,
      batchSize: 64,
      scheduler: 'linear',
      ewcLambda: 2000,
    });
  }
}
