//! Recurrent-Depth Transformer (RDT) execution (ADR-latest)
//!
//! Standard transformers map depth linearly to parameter count: a 32-layer
//! model carries 32 distinct weight sets. A **Recurrent-Depth Transformer**
//! shares a single block of weights (or a small set of blocks) and routes the
//! hidden state through them repeatedly, forming a deep computational loop in
//! latent space. This lets the model reason *deeper* without growing *larger*.
//!
//! Looping a fixed number of times wastes compute, so this module implements an
//! **Adaptive Halting Mechanism** (PonderNet / Universal-Transformer style). A
//! lightweight linear probe evaluates the hidden state after every loop and
//! decides, per token, whether that token is "cooked" enough to exit. Easy
//! tokens (`"the"`) resolve in a couple of loops; hard tokens (a logic gate in
//! code) may run to the `max_loops` ceiling.
//!
//! # The Honest Boundary
//!
//! This path is **only** valid for weights natively trained for weight-sharing
//! (ALBERT-style cross-layer sharing or an explicit RDT fine-tune). Running
//! standard Llama/Qwen weights through a shared block emits garbage tokens
//! because those weights were trained for distinct per-layer transforms. The
//! loader therefore validates GGUF metadata on initialization and refuses
//! incompatible weights via [`validate_rdt_metadata`] rather than silently
//! producing nonsense. See [`RdtCompatibilityError`].
//!
//! # Telemetry
//!
//! Inference latency is no longer deterministic per token. The realized loop
//! depth is recorded in [`DepthTelemetry`] (mean / max / per-call history) so an
//! audit dashboard or evolutionary harness can track `mean_inference_depth` and
//! penalize agents that pick overly expensive RDT configs for trivial work.
//!
//! # Substrate status
//!
//! The execution graph is fully implemented and unit-tested with synthetic
//! weights. End-to-end generation is dormant until a compatible RDT GGUF is
//! supplied; see [`validate_rdt_metadata`].

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::error::{Result, RuvLLMError};

// ---------------------------------------------------------------------------
// Configuration (always available, no candle dependency)
// ---------------------------------------------------------------------------

/// Configuration for a Recurrent-Depth Transformer.
///
/// Mirrors the subset of GGUF metadata needed to build the shared-block
/// execution graph plus the recurrent-loop controls (`max_loops`,
/// `halt_threshold`).
#[derive(Debug, Clone, PartialEq)]
pub struct RdtConfig {
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Number of attention query heads.
    pub num_heads: usize,
    /// Number of key/value heads (GQA). Equal to `num_heads` for MHA.
    pub num_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length (for RoPE table sizing).
    pub max_position_embeddings: usize,
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Number of *distinct* shared blocks. `1` is the canonical RDT; a small
    /// value (e.g. 2) supports ALBERT-style grouped sharing.
    pub num_shared_blocks: usize,
    /// Maximum recurrent loop iterations (the depth ceiling).
    pub max_loops: usize,
    /// Halting probability threshold in `(0, 1]`. A token exits once its
    /// `p_halt` reaches this value.
    pub halt_threshold: f32,
}

impl Default for RdtConfig {
    fn default() -> Self {
        // A small, valid config suitable for tests and smoke runs.
        Self {
            hidden_size: 256,
            intermediate_size: 688,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1024,
            max_position_embeddings: 2048,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_shared_blocks: 1,
            max_loops: 16,
            halt_threshold: 0.9,
        }
    }
}

impl RdtConfig {
    /// Head dimension (`hidden_size / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// GQA grouping ratio (`num_heads / num_kv_heads`).
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }

    /// Validate structural invariants of the configuration.
    ///
    /// This guards the *shape* contract; semantic RDT-compatibility of weights
    /// is enforced separately by [`validate_rdt_metadata`].
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(RuvLLMError::Config(
                "RDT: hidden_size and num_heads must be non-zero".into(),
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(RuvLLMError::Config(format!(
                "RDT: hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(RuvLLMError::Config(format!(
                "RDT: num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.num_shared_blocks == 0 {
            return Err(RuvLLMError::Config(
                "RDT: num_shared_blocks must be >= 1".into(),
            ));
        }
        if self.max_loops == 0 {
            return Err(RuvLLMError::Config("RDT: max_loops must be >= 1".into()));
        }
        if !(self.halt_threshold > 0.0 && self.halt_threshold <= 1.0) {
            return Err(RuvLLMError::Config(format!(
                "RDT: halt_threshold ({}) must be in (0, 1]",
                self.halt_threshold
            )));
        }
        if self.max_loops < self.num_shared_blocks {
            return Err(RuvLLMError::Config(format!(
                "RDT: max_loops ({}) must be >= num_shared_blocks ({})",
                self.max_loops, self.num_shared_blocks
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The Honest Boundary: weight-sharing compatibility validation
// ---------------------------------------------------------------------------

/// Error returned when a model is asked to run through the RDT path but its
/// weights were not trained for weight-sharing.
///
/// Surfacing this as a hard error (rather than emitting garbage tokens) is the
/// "honest boundary" mandated by the ADR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdtCompatibilityError {
    /// The GGUF `general.architecture` we observed.
    pub detected_architecture: String,
    /// Human-readable reason the weights were rejected.
    pub reason: String,
}

impl std::fmt::Display for RdtCompatibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "non-RDT GGUF rejected (architecture='{}'): {}. \
             The RDT execution path requires weights natively trained for \
             weight-sharing (ALBERT-style cross-layer sharing or an explicit \
             RDT fine-tune). Running standard weights through a shared block \
             produces garbage tokens.",
            self.detected_architecture, self.reason
        )
    }
}

impl std::error::Error for RdtCompatibilityError {}

impl From<RdtCompatibilityError> for RuvLLMError {
    fn from(e: RdtCompatibilityError) -> Self {
        RuvLLMError::Model(e.to_string())
    }
}

/// GGUF metadata keys that mark a checkpoint as weight-sharing / RDT.
///
/// A compatible export must set `general.architecture` to one of
/// [`RDT_ARCHITECTURES`] *or* declare an explicit recurrence flag via one of
/// these keys.
pub const RDT_RECURRENCE_KEYS: &[&str] = &[
    "rdt.recurrent",
    "rdt.weight_sharing",
    "general.weight_sharing",
    "recurrent_depth.enabled",
];

/// `general.architecture` values recognized as natively weight-sharing.
pub const RDT_ARCHITECTURES: &[&str] =
    &["rdt", "recurrent_depth", "albert", "universal_transformer"];

/// Validate that GGUF metadata describes a weight-sharing (RDT) checkpoint.
///
/// This is the load-time gate enforcing the honest boundary. It accepts the
/// metadata as a string map (the architecture-agnostic projection of GGUF
/// metadata) so it can be unit-tested without a real GGUF file and reused from
/// the candle loader.
///
/// A checkpoint is accepted iff **either**:
/// - `general.architecture` is one of [`RDT_ARCHITECTURES`], or
/// - one of [`RDT_RECURRENCE_KEYS`] is present and truthy
///   (`true` / `1` / `yes`, case-insensitive).
///
/// # Errors
///
/// Returns [`RdtCompatibilityError`] for any non-RDT architecture (e.g.
/// `llama`, `qwen2`). Per the ADR the caller should treat this as fatal at
/// initialization rather than proceeding to emit garbage tokens.
pub fn validate_rdt_metadata(
    metadata: &BTreeMap<String, String>,
) -> std::result::Result<(), RdtCompatibilityError> {
    let arch = metadata
        .get("general.architecture")
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    if RDT_ARCHITECTURES.contains(&arch.as_str()) {
        return Ok(());
    }

    for key in RDT_RECURRENCE_KEYS {
        if let Some(raw) = metadata.get(*key) {
            if is_truthy(raw) {
                return Ok(());
            }
        }
    }

    let reason = if arch.is_empty() {
        "no 'general.architecture' and no recurrence flag found".to_string()
    } else {
        format!(
            "architecture '{}' is not weight-sharing and no recurrence flag \
             ({:?}) was set",
            arch, RDT_RECURRENCE_KEYS
        )
    };

    Err(RdtCompatibilityError {
        detected_architecture: if arch.is_empty() {
            "<unknown>".to_string()
        } else {
            arch
        },
        reason,
    })
}

fn is_truthy(raw: &str) -> bool {
    matches!(
        raw.trim().to_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

// ---------------------------------------------------------------------------
// Depth telemetry (always available)
// ---------------------------------------------------------------------------

/// Snapshot of recurrent-depth statistics over recorded forward passes.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DepthStats {
    /// Number of recorded forward passes.
    pub samples: usize,
    /// Mean realized loop depth across all recorded tokens.
    pub mean_inference_depth: f32,
    /// Maximum loop depth observed in any single pass.
    pub max_inference_depth: usize,
    /// Minimum loop depth observed in any single pass.
    pub min_inference_depth: usize,
}

impl Default for DepthStats {
    fn default() -> Self {
        Self {
            samples: 0,
            mean_inference_depth: 0.0,
            max_inference_depth: 0,
            min_inference_depth: 0,
        }
    }
}

#[derive(Debug, Default)]
struct DepthInner {
    /// Per-call mean token depth.
    means: Vec<f32>,
    /// Per-call maximum token depth.
    maxes: Vec<usize>,
    /// Per-call minimum token depth.
    mins: Vec<usize>,
}

/// Thread-safe recorder for per-token recurrent depth.
///
/// `bench/system-audit.mjs` consumes `mean_inference_depth` from
/// [`DepthStats`] to track token-efficiency over time.
#[derive(Debug, Default)]
pub struct DepthTelemetry {
    inner: Mutex<DepthInner>,
}

impl DepthTelemetry {
    /// Create an empty telemetry recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one forward pass given the per-token realized depths.
    pub fn record(&self, token_depths: &[usize]) {
        if token_depths.is_empty() {
            return;
        }
        let sum: usize = token_depths.iter().sum();
        let mean = sum as f32 / token_depths.len() as f32;
        let max = *token_depths.iter().max().unwrap();
        let min = *token_depths.iter().min().unwrap();

        let mut inner = self.inner.lock().unwrap();
        inner.means.push(mean);
        inner.maxes.push(max);
        inner.mins.push(min);
    }

    /// Aggregate statistics over all recorded passes.
    pub fn stats(&self) -> DepthStats {
        let inner = self.inner.lock().unwrap();
        let samples = inner.means.len();
        if samples == 0 {
            return DepthStats::default();
        }
        let mean = inner.means.iter().sum::<f32>() / samples as f32;
        DepthStats {
            samples,
            mean_inference_depth: mean,
            max_inference_depth: inner.maxes.iter().copied().max().unwrap_or(0),
            min_inference_depth: inner.mins.iter().copied().min().unwrap_or(0),
        }
    }

    /// Reset all recorded telemetry.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.means.clear();
        inner.maxes.clear();
        inner.mins.clear();
    }

    /// Serialize the aggregate [`DepthStats`] to a JSON string for audit
    /// dashboards (`bench/system-audit.mjs` tracks `mean_inference_depth`).
    pub fn report_json(&self) -> String {
        serde_json::to_string(&self.stats()).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// Candle execution graph
// ---------------------------------------------------------------------------

#[cfg(feature = "candle")]
pub use candle_impl::{HaltingRouter, RdtCache, RdtModel, SharedBlock};

#[cfg(feature = "candle")]
mod candle_impl {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor, D};
    use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

    /// The routing head that decides whether a hidden state is "cooked" enough
    /// to halt. Projects the hidden state to a per-token scalar and applies a
    /// sigmoid to obtain `p_halt`.
    pub struct HaltingRouter {
        proj: Linear,
        threshold: f32,
    }

    impl HaltingRouter {
        /// Build from a projection layer and halting threshold.
        pub fn new(proj: Linear, threshold: f32) -> Self {
            Self { proj, threshold }
        }

        /// Load the routing head from a [`VarBuilder`].
        pub fn load(vb: VarBuilder, hidden_size: usize, threshold: f32) -> Result<Self> {
            // [hidden_size] -> [1]; bias lets the probe learn a base halt rate.
            let proj = candle_nn::linear(hidden_size, 1, vb.pp("proj")).map_err(cand)?;
            Ok(Self::new(proj, threshold))
        }

        /// Compute halting probability for each token.
        ///
        /// Returns `p_halt` shaped `[batch, seq, 1]`.
        pub fn p_halt(&self, hidden_state: &Tensor) -> Result<Tensor> {
            let logits = self.proj.forward(hidden_state).map_err(cand)?;
            ops::sigmoid(&logits).map_err(cand)
        }

        /// Convenience: `(p_halt, should_stop)` using the batch-max policy from
        /// the ADR. The full per-token policy lives in
        /// [`RdtModel::forward`]; this mirrors the ADR's reference signature and
        /// is handy for diagnostics.
        pub fn compute_halt(&self, hidden_state: &Tensor) -> Result<(Tensor, bool)> {
            let p_halt = self.p_halt(hidden_state)?;
            let max_p = p_halt
                .max_all()
                .map_err(cand)?
                .to_scalar::<f32>()
                .map_err(cand)?;
            Ok((p_halt, max_p >= self.threshold))
        }
    }

    /// A single shared transformer block: pre-norm attention + pre-norm SwiGLU
    /// MLP, each with a residual connection. The *same* block instance is
    /// applied repeatedly by [`RdtModel`].
    pub struct SharedBlock {
        input_norm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        post_attn_norm: RmsNorm,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    }

    impl SharedBlock {
        /// Load a shared block from a [`VarBuilder`].
        pub fn load(vb: VarBuilder, cfg: &RdtConfig) -> Result<Self> {
            let h = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let q_out = cfg.num_heads * head_dim;
            let kv_out = cfg.num_kv_heads * head_dim;

            let input_norm =
                candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm")).map_err(cand)?;
            let attn = vb.pp("self_attn");
            let q_proj = candle_nn::linear_no_bias(h, q_out, attn.pp("q_proj")).map_err(cand)?;
            let k_proj = candle_nn::linear_no_bias(h, kv_out, attn.pp("k_proj")).map_err(cand)?;
            let v_proj = candle_nn::linear_no_bias(h, kv_out, attn.pp("v_proj")).map_err(cand)?;
            let o_proj = candle_nn::linear_no_bias(q_out, h, attn.pp("o_proj")).map_err(cand)?;

            let post_attn_norm =
                candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))
                    .map_err(cand)?;
            let mlp = vb.pp("mlp");
            let gate_proj =
                candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp.pp("gate_proj"))
                    .map_err(cand)?;
            let up_proj = candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp.pp("up_proj"))
                .map_err(cand)?;
            let down_proj =
                candle_nn::linear_no_bias(cfg.intermediate_size, h, mlp.pp("down_proj"))
                    .map_err(cand)?;

            Ok(Self {
                input_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                post_attn_norm,
                gate_proj,
                up_proj,
                down_proj,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim,
            })
        }

        /// One pass of the shared block over `[batch, seq, hidden]`.
        ///
        /// `cos`/`sin` are the RoPE tables for the current positions, and
        /// `mask` is the additive causal mask `[seq, seq]`.
        pub fn forward(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
        ) -> Result<Tensor> {
            let (out, _kv) = self.forward_cached(xs, cos, sin, mask, None)?;
            Ok(out)
        }

        /// Cached pass: `past` is the prior (rotated) K/V; returns the block
        /// output and the *full* (past + current) K/V for the caller to store.
        pub fn forward_cached(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&(Tensor, Tensor)>,
        ) -> Result<(Tensor, (Tensor, Tensor))> {
            let (b, seq, _h) = xs.dims3().map_err(cand)?;

            // --- Attention sub-layer (pre-norm + residual) ---
            let normed = self.input_norm.forward(xs).map_err(cand)?;
            let (attn_out, kv) = self.attention(&normed, cos, sin, mask, past, b, seq)?;
            let xs = (xs + attn_out).map_err(cand)?;

            // --- MLP sub-layer (pre-norm + residual) ---
            let normed = self.post_attn_norm.forward(&xs).map_err(cand)?;
            let mlp_out = self.mlp(&normed)?;
            let out = (xs + mlp_out).map_err(cand)?;
            Ok((out, kv))
        }

        fn attention(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&(Tensor, Tensor)>,
            b: usize,
            seq: usize,
        ) -> Result<(Tensor, (Tensor, Tensor))> {
            let q = self.q_proj.forward(xs).map_err(cand)?;
            let k = self.k_proj.forward(xs).map_err(cand)?;
            let v = self.v_proj.forward(xs).map_err(cand)?;

            // [b, seq, n*hd] -> [b, n, seq, hd]
            let q = q
                .reshape((b, seq, self.num_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;
            let k = k
                .reshape((b, seq, self.num_kv_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;
            let v = v
                .reshape((b, seq, self.num_kv_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;

            let q = apply_rope(&q, cos, sin)?;
            let k_cur = apply_rope(&k, cos, sin)?;

            // Concatenate with past (already-rotated) keys/values.
            let (k_full, v_full) = match past {
                Some((pk, pv)) => (
                    Tensor::cat(&[pk, &k_cur], 2).map_err(cand)?,
                    Tensor::cat(&[pv, &v], 2).map_err(cand)?,
                ),
                None => (k_cur, v),
            };

            // GQA: repeat kv heads to match query heads.
            let k = repeat_kv(&k_full, self.num_heads / self.num_kv_heads)?;
            let v = repeat_kv(&v_full, self.num_heads / self.num_kv_heads)?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let scores = (q.matmul(&k.transpose(2, 3).map_err(cand)?).map_err(cand)? * scale)
                .map_err(cand)?;
            // Additive causal mask broadcast over [b, n, seq, kv_len].
            let scores = scores.broadcast_add(mask).map_err(cand)?;
            let probs = ops::softmax_last_dim(&scores).map_err(cand)?;

            let ctx = probs.matmul(&v).map_err(cand)?; // [b, n, seq, hd]
            let ctx = ctx
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?
                .reshape((b, seq, self.num_heads * self.head_dim))
                .map_err(cand)?;
            let out = self.o_proj.forward(&ctx).map_err(cand)?;
            Ok((out, (k_full, v_full)))
        }

        fn mlp(&self, xs: &Tensor) -> Result<Tensor> {
            let gate = self.gate_proj.forward(xs).map_err(cand)?;
            let gate = ops::silu(&gate).map_err(cand)?;
            let up = self.up_proj.forward(xs).map_err(cand)?;
            let hidden = (gate * up).map_err(cand)?;
            self.down_proj.forward(&hidden).map_err(cand)
        }
    }

    /// The Recurrent-Depth Transformer model.
    pub struct RdtModel {
        embed_tokens: Embedding,
        /// One or more shared blocks, cycled through across loop iterations.
        shared_blocks: Vec<SharedBlock>,
        halting_router: HaltingRouter,
        ln_f: RmsNorm,
        lm_head: Linear,
        cfg: RdtConfig,
        device: Device,
        dtype: DType,
        /// Recurrent-depth telemetry (see [`DepthTelemetry`]).
        pub telemetry: DepthTelemetry,
    }

    impl RdtModel {
        /// Build an RDT model from a [`VarBuilder`].
        ///
        /// Validates [`RdtConfig`] structurally. Weight-sharing compatibility of
        /// the *source checkpoint* must be enforced separately by the loader via
        /// [`validate_rdt_metadata`] — that is the honest boundary.
        pub fn load(vb: VarBuilder, cfg: RdtConfig) -> Result<Self> {
            cfg.validate()?;
            let device = vb.device().clone();
            let dtype = vb.dtype();

            let embed_tokens =
                candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                    .map_err(cand)?;

            let mut shared_blocks = Vec::with_capacity(cfg.num_shared_blocks);
            let blocks_vb = vb.pp("shared_blocks");
            for i in 0..cfg.num_shared_blocks {
                shared_blocks.push(SharedBlock::load(blocks_vb.pp(i), &cfg)?);
            }

            let halting_router =
                HaltingRouter::load(vb.pp("halting_router"), cfg.hidden_size, cfg.halt_threshold)?;
            let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
                .map_err(cand)?;
            let lm_head =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
                    .map_err(cand)?;

            Ok(Self {
                embed_tokens,
                shared_blocks,
                halting_router,
                ln_f,
                lm_head,
                cfg,
                device,
                dtype,
                telemetry: DepthTelemetry::new(),
            })
        }

        /// Access the model configuration.
        pub fn config(&self) -> &RdtConfig {
            &self.cfg
        }

        /// Run a forward pass over `input_ids` (`[batch, seq]`, dtype u32),
        /// returning logits `[batch, seq, vocab]`.
        ///
        /// Internally embeds the tokens and drives the recurrent loop with
        /// per-token adaptive halting before the final norm + LM head.
        pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
            let mut cache = RdtCache::new();
            self.forward_cached(input_ids, &mut cache)
        }

        /// Forward that reads and updates `cache` for incremental decode.
        /// Processes the `seq` new positions in `input_ids` against the cached
        /// prefix and returns logits `[batch, seq, vocab]`.
        pub fn forward_cached(&self, input_ids: &Tensor, cache: &mut RdtCache) -> Result<Tensor> {
            let (b, seq) = input_ids.dims2().map_err(cand)?;
            let offset = cache.seq_len;
            let xs = self.embed_tokens.forward(input_ids).map_err(cand)?;
            let xs = xs.to_dtype(self.dtype).map_err(cand)?;

            let (cos, sin) = self.rope_tables(seq, offset)?;
            let mask = self.causal_mask(seq, offset + seq, offset)?;

            let (hidden, kv) =
                self.recurrent_loop(&xs, &cos, &sin, &mask, cache.kv.as_ref(), b, seq)?;
            cache.kv = Some(kv);
            cache.seq_len += seq;

            let normed = self.ln_f.forward(&hidden).map_err(cand)?;
            self.lm_head.forward(&normed).map_err(cand)
        }

        /// Greedy autoregressive generation from a single-sequence prompt.
        /// Returns the newly generated token ids; stops early on `eos`.
        pub fn generate(
            &self,
            prompt_ids: &[u32],
            max_new_tokens: usize,
            eos: Option<u32>,
        ) -> Result<Vec<u32>> {
            if prompt_ids.is_empty() {
                return Err(RuvLLMError::Generation("empty prompt".into()));
            }
            let mut cache = RdtCache::new();
            let prompt = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &self.device)
                .map_err(cand)?;
            let logits = self.forward_cached(&prompt, &mut cache)?;
            let mut next = last_argmax(&logits)?;

            let mut out = Vec::with_capacity(max_new_tokens);
            for _ in 0..max_new_tokens {
                out.push(next);
                if Some(next) == eos {
                    break;
                }
                let step = Tensor::from_vec(vec![next], (1, 1), &self.device).map_err(cand)?;
                let logits = self.forward_cached(&step, &mut cache)?;
                next = last_argmax(&logits)?;
            }
            Ok(out)
        }

        /// The deep recurrent loop with per-token adaptive halting.
        ///
        /// Each iteration applies a shared block, then the routing head emits a
        /// per-token `p_halt`. Tokens crossing `halt_threshold` halt: their
        /// hidden state is frozen for subsequent iterations while still-running
        /// tokens keep computing. The loop ends when every token has halted or
        /// `max_loops` is reached. Returns the final hidden state and the
        /// final-iteration K/V (for the decode cache); records depth in telemetry.
        fn recurrent_loop(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&(Tensor, Tensor)>,
            b: usize,
            seq: usize,
        ) -> Result<(Tensor, (Tensor, Tensor))> {
            let n = b * seq;
            let mut hidden = xs.clone();

            // GPU-resident ACT state — eliminates per-iteration to_vec1()/from_vec() transfers.
            // `running_f32`:  1.0 for still-running tokens, 0.0 for halted, [b, seq, 1]
            // `depth_f32`:    count of iterations each token was running, [b, seq, 1]
            let mut running_f32 =
                Tensor::ones((b, seq, 1), DType::F32, &self.device).map_err(cand)?;
            let mut depth_f32 =
                Tensor::zeros((b, seq, 1), DType::F32, &self.device).map_err(cand)?;
            let mut last_kv: Option<(Tensor, Tensor)> = None;

            let max_loops = self.cfg.max_loops;
            for step in 0..max_loops {
                let block = &self.shared_blocks[step % self.shared_blocks.len()];
                let (candidate, kv) = block.forward_cached(&hidden, cos, sin, mask, past)?;
                last_kv = Some(kv);

                // Freeze halted tokens: hidden = running*candidate + (1-running)*hidden.
                let running_typed = running_f32.to_dtype(self.dtype).map_err(cand)?;
                let halted_typed =
                    (running_typed.ones_like().map_err(cand)? - &running_typed).map_err(cand)?;
                hidden = (candidate.broadcast_mul(&running_typed).map_err(cand)?
                    + hidden.broadcast_mul(&halted_typed).map_err(cand)?)
                .map_err(cand)?;

                // Depth: increment per-token count for tokens that were running this step.
                depth_f32 = (&depth_f32 + &running_f32).map_err(cand)?;

                // Halting decision — fully on device, no weight-vector transfer.
                let p_halt = self.halting_router.p_halt(&hidden)?;
                let p_halt_f32 = p_halt.to_dtype(DType::F32).map_err(cand)?;
                let should_halt = p_halt_f32
                    .ge(self.cfg.halt_threshold as f64)
                    .map_err(cand)?
                    .to_dtype(DType::F32)
                    .map_err(cand)?;
                // Newly halted = was running AND p_halt >= threshold.
                let newly_halted = (&should_halt * &running_f32).map_err(cand)?;
                running_f32 = (&running_f32 - &newly_halted).map_err(cand)?;

                tracing::trace!(step, "rdt loop iteration");

                // Early-exit: one scalar sync (cheap vs. the eliminated weight-vector transfers).
                let any_running = running_f32
                    .sum_all()
                    .map_err(cand)?
                    .to_scalar::<f32>()
                    .map_err(cand)?
                    > 0.5;
                if !any_running {
                    break;
                }
            }

            // Single depth sync at the end (not in the hot path).
            let depth_vec: Vec<f32> = depth_f32
                .reshape((n,))
                .map_err(cand)?
                .to_vec1()
                .map_err(cand)?;
            let depth: Vec<usize> = depth_vec.into_iter().map(|d| d as usize).collect();
            self.telemetry.record(&depth);

            let kv = last_kv.expect("at least one loop iteration runs");
            Ok((hidden, kv))
        }

        /// RoPE cos/sin tables for `seq` positions at absolute offset
        /// `offset..offset+seq`: `[seq, head_dim]`.
        fn rope_tables(&self, seq: usize, offset: usize) -> Result<(Tensor, Tensor)> {
            let head_dim = self.cfg.head_dim();
            let half = head_dim / 2;
            let theta = self.cfg.rope_theta as f64;

            let inv_freq: Vec<f32> = (0..half)
                .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
                .collect();
            let inv_freq = Tensor::from_vec(inv_freq, (1, half), &self.device).map_err(cand)?;
            let positions: Vec<f32> = (0..seq).map(|p| (p + offset) as f32).collect();
            let positions = Tensor::from_vec(positions, (seq, 1), &self.device).map_err(cand)?;

            // [seq, half]
            let freqs = positions.matmul(&inv_freq).map_err(cand)?;
            // Duplicate to full head_dim: [seq, head_dim].
            let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(cand)?;
            let cos = freqs
                .cos()
                .map_err(cand)?
                .to_dtype(self.dtype)
                .map_err(cand)?;
            let sin = freqs
                .sin()
                .map_err(cand)?
                .to_dtype(self.dtype)
                .map_err(cand)?;
            Ok((cos, sin))
        }

        /// Additive causal mask `[q_len, kv_len]`. Query `i` (absolute
        /// `offset + i`) may attend to key positions `<= offset + i`.
        fn causal_mask(&self, q_len: usize, kv_len: usize, offset: usize) -> Result<Tensor> {
            let mut data = vec![0f32; q_len * kv_len];
            for i in 0..q_len {
                let allowed = offset + i;
                for j in 0..kv_len {
                    if j > allowed {
                        data[i * kv_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            Tensor::from_vec(data, (q_len, kv_len), &self.device)
                .map_err(cand)?
                .to_dtype(self.dtype)
                .map_err(cand)
        }
    }

    /// Incremental KV cache for RDT decode (final-iteration K/V of the shared
    /// block, concatenated across decode steps).
    #[derive(Default)]
    pub struct RdtCache {
        kv: Option<(Tensor, Tensor)>,
        seq_len: usize,
    }

    impl RdtCache {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn len(&self) -> usize {
            self.seq_len
        }
        pub fn is_empty(&self) -> bool {
            self.seq_len == 0
        }
        pub fn reset(&mut self) {
            self.kv = None;
            self.seq_len = 0;
        }
    }

    /// Argmax over vocab at the last sequence position of `[1, seq, vocab]`.
    fn last_argmax(logits: &Tensor) -> Result<u32> {
        let (_b, seq, _v) = logits.dims3().map_err(cand)?;
        let last = logits.i((0, seq - 1)).map_err(cand)?;
        let row: Vec<f32> = last
            .to_dtype(DType::F32)
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)?;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best as u32)
    }

    /// Apply rotary position embeddings to `[b, n, seq, head_dim]`.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // cos/sin: [seq, head_dim] -> broadcast over [b, n, seq, head_dim].
        let (_b, _n, seq, hd) = x.dims4().map_err(cand)?;
        let cos = cos
            .narrow(0, 0, seq)
            .map_err(cand)?
            .reshape((1, 1, seq, hd))
            .map_err(cand)?;
        let sin = sin
            .narrow(0, 0, seq)
            .map_err(cand)?
            .reshape((1, 1, seq, hd))
            .map_err(cand)?;
        let rotated = rotate_half(x)?;
        let out = (x.broadcast_mul(&cos).map_err(cand)?
            + rotated.broadcast_mul(&sin).map_err(cand)?)
        .map_err(cand)?;
        Ok(out)
    }

    /// `rotate_half([x1, x2]) = [-x2, x1]` along the last dimension.
    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let hd = x.dim(D::Minus1).map_err(cand)?;
        let half = hd / 2;
        let x1 = x.narrow(D::Minus1, 0, half).map_err(cand)?;
        let x2 = x.narrow(D::Minus1, half, hd - half).map_err(cand)?;
        let neg_x2 = x2.neg().map_err(cand)?;
        Tensor::cat(&[&neg_x2, &x1], D::Minus1).map_err(cand)
    }

    /// Repeat KV heads `n_rep` times for grouped-query attention.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv_heads, seq, hd) = x.dims4().map_err(cand)?;
        // [b, kv, seq, hd] -> [b, kv, 1, seq, hd] -> expand -> [b, kv*n_rep, seq, hd]
        x.unsqueeze(2)
            .map_err(cand)?
            .expand((b, kv_heads, n_rep, seq, hd))
            .map_err(cand)?
            .reshape((b, kv_heads * n_rep, seq, hd))
            .map_err(cand)
    }

    /// Map a candle error into the crate error type.
    fn cand(e: candle_core::Error) -> RuvLLMError {
        RuvLLMError::Model(format!("candle (rdt): {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn config_default_is_valid() {
        assert!(RdtConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_bad_shapes() {
        let mut c = RdtConfig::default();
        c.num_heads = 7; // 256 % 7 != 0
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.halt_threshold = 1.5;
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.max_loops = 0;
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.num_heads = 8;
        c.num_kv_heads = 3; // 8 % 3 != 0
        assert!(c.validate().is_err());
    }

    #[test]
    fn honest_boundary_rejects_llama() {
        let m = meta(&[("general.architecture", "llama")]);
        let err = validate_rdt_metadata(&m).unwrap_err();
        assert_eq!(err.detected_architecture, "llama");
        assert!(err.to_string().contains("garbage tokens"));
    }

    #[test]
    fn honest_boundary_rejects_qwen2() {
        let m = meta(&[("general.architecture", "qwen2")]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn honest_boundary_rejects_missing_metadata() {
        let m = meta(&[]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn honest_boundary_accepts_rdt_architecture() {
        for arch in RDT_ARCHITECTURES {
            let m = meta(&[("general.architecture", arch)]);
            assert!(validate_rdt_metadata(&m).is_ok(), "arch {arch} should pass");
        }
    }

    #[test]
    fn honest_boundary_accepts_recurrence_flag() {
        // A llama-derived RDT fine-tune declares recurrence explicitly.
        let m = meta(&[("general.architecture", "llama"), ("rdt.recurrent", "true")]);
        assert!(validate_rdt_metadata(&m).is_ok());

        let m = meta(&[("recurrent_depth.enabled", "1")]);
        assert!(validate_rdt_metadata(&m).is_ok());
    }

    #[test]
    fn honest_boundary_rejects_falsey_flag() {
        let m = meta(&[
            ("general.architecture", "llama"),
            ("rdt.recurrent", "false"),
        ]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn telemetry_aggregates() {
        let t = DepthTelemetry::new();
        t.record(&[2, 4, 6]); // mean 4, max 6, min 2
        t.record(&[1, 1, 1]); // mean 1, max 1, min 1
        let s = t.stats();
        assert_eq!(s.samples, 2);
        assert_eq!(s.max_inference_depth, 6);
        assert_eq!(s.min_inference_depth, 1);
        assert!((s.mean_inference_depth - 2.5).abs() < 1e-6);

        t.reset();
        assert_eq!(t.stats().samples, 0);
    }

    #[test]
    fn telemetry_ignores_empty() {
        let t = DepthTelemetry::new();
        t.record(&[]);
        assert_eq!(t.stats().samples, 0);
    }

    // ---- candle-backed execution tests (synthetic weights) ----

    #[cfg(feature = "candle")]
    mod candle_tests {
        use super::*;
        use candle_core::{DType, Device, IndexOp, Tensor};
        use candle_nn::VarBuilder;

        fn tiny_cfg() -> RdtConfig {
            RdtConfig {
                hidden_size: 32,
                intermediate_size: 64,
                num_heads: 4,
                num_kv_heads: 2,
                vocab_size: 48,
                max_position_embeddings: 64,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                num_shared_blocks: 1,
                max_loops: 8,
                halt_threshold: 0.9,
            }
        }

        #[test]
        fn forward_shapes_are_correct() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            // Zeros builder fabricates any requested tensor as zeros — enough to
            // exercise the full execution graph deterministically.
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).expect("load");

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (1, 5), &dev).unwrap();
            let logits = model.forward(&input_ids).expect("forward");
            assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        }

        #[test]
        fn zero_router_runs_to_max_loops() {
            // With zero weights, p_halt = sigmoid(0) = 0.5 < 0.9 threshold, so
            // every token runs to the ceiling. This proves the loop honors
            // max_loops and records depth telemetry.
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
            let _ = model.forward(&input_ids).unwrap();

            let stats = model.telemetry.stats();
            assert_eq!(stats.samples, 1);
            assert_eq!(stats.max_inference_depth, cfg.max_loops);
            assert_eq!(stats.min_inference_depth, cfg.max_loops);
        }

        #[test]
        fn batched_forward_works() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
            assert_eq!(model.telemetry.stats().samples, 1);
        }

        #[test]
        fn multi_block_sharing_loads_and_runs() {
            let dev = Device::Cpu;
            let mut cfg = tiny_cfg();
            cfg.num_shared_blocks = 2; // ALBERT-style grouped sharing
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();
            let input_ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
        }

        #[test]
        fn forward_output_is_finite() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg).unwrap();
            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
            assert!(flat.iter().all(|x| x.is_finite()), "logits must be finite");
        }

        #[test]
        fn generate_produces_tokens() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();
            let out = model.generate(&[1, 2, 3], 5, None).unwrap();
            assert_eq!(out.len(), 5);
            assert!(out.iter().all(|&t| (t as usize) < cfg.vocab_size));
        }

        #[test]
        fn cached_decode_matches_full_at_single_loop() {
            // At max_loops=1 the recurrent caching is exact (single iteration),
            // so incremental decode must match a full forward. Use random
            // weights so the check is non-degenerate.
            use candle_nn::VarMap;
            let dev = Device::Cpu;
            let mut cfg = tiny_cfg();
            cfg.max_loops = 1;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let ids = vec![3u32, 7, 1, 9, 4];
            let full_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev).unwrap();
            let full = model.forward(&full_ids).unwrap();
            let full_last: Vec<f32> = full.i((0, ids.len() - 1)).unwrap().to_vec1().unwrap();

            let mut cache = RdtCache::new();
            let mut last: Vec<f32> = vec![];
            for (k, &tok) in ids.iter().enumerate() {
                let step = Tensor::from_vec(vec![tok], (1, 1), &dev).unwrap();
                let logits = model.forward_cached(&step, &mut cache).unwrap();
                assert_eq!(cache.len(), k + 1);
                last = logits.i((0, 0)).unwrap().to_vec1().unwrap();
            }
            let max_diff = full_last
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_diff < 1e-3, "RDT KV-cache decode diverged: {max_diff}");
        }

        #[test]
        fn telemetry_report_json_roundtrips() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg).unwrap();
            let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
            let _ = model.forward(&ids).unwrap();
            let json = model.telemetry.report_json();
            assert!(json.contains("mean_inference_depth"));
            let parsed: DepthStats = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.samples, 1);
        }
    }
}
