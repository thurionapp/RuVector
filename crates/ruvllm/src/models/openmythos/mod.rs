//! OpenMythos — a Recurrent-Depth Transformer in Rust/Candle.
//!
//! A faithful port of <https://github.com/kyegomez/OpenMythos> (an open
//! reconstruction of a recurrent-depth "Mythos"-style model), built on the RDT
//! substrate in [`crate::models::rdt`]. The distinctive OpenMythos design is
//! captured in full:
//!
//! - **Prelude → Recurrent → Coda** staging ([`OpenMythos::forward`]).
//! - **LTI-constrained injection** `h_{t+1} = A·h_t + B·e + Transformer(h_t + e)`
//!   with contractive diagonal `A = exp(-exp(log_dt + log_A)) ∈ (0,1)`
//!   ([`recurrent::LtiInjection`]).
//! - **Adaptive Computation Time** halting with remainder weighting and a
//!   **loop-index positional embedding** ([`recurrent::RecurrentBlock`]).
//! - **Attention variants**: Grouped-Query and Multi-Latent (compressed KV),
//!   both with incremental **KV-cache decode** ([`attention`]).
//! - **Per-depth LoRA** adaptation in the recurrent loop
//!   ([`recurrent::DepthLora`]).
//! - A **checkpoint loader** with the honest-boundary metadata gate
//!   ([`validate_mythos_metadata`]).
//!
//! Real generation is dormant until a compatible checkpoint is supplied; the
//! execution graph (including greedy [`OpenMythos::generate`]) is exercised with
//! synthetic weights in the unit tests.

#![cfg(feature = "candle")]

mod attention;
mod block;
mod config;
mod ffn;

#[cfg(all(feature = "cuda", feature = "fused-act"))]
pub mod act_kernel;
mod recurrent;
mod rope;

pub use attention::KvLayerCache;
pub use config::{
    validate_mythos_metadata, AttnType, MythosCompatibilityError, MythosConfig,
    MYTHOS_ARCHITECTURES, MYTHOS_RECURRENCE_KEYS,
};

use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

use crate::error::{Result, RuvLLMError};
use crate::models::rdt::DepthTelemetry;
use crate::models::sampling::{Sampler, SamplingConfig};

use block::TransformerBlock;
use recurrent::RecurrentBlock;
use rope::{cand, causal_mask, rope_tables};

/// Incremental KV cache spanning all attention layers (prelude, recurrent,
/// coda). One instance is threaded through a decode session.
pub struct MythosCache {
    prelude: Vec<Option<KvLayerCache>>,
    recurrent: Option<KvLayerCache>,
    coda: Vec<Option<KvLayerCache>>,
    /// Number of positions already processed (the RoPE / mask offset).
    seq_len: usize,
}

impl MythosCache {
    /// Empty cache sized for `cfg`.
    pub fn new(cfg: &MythosConfig) -> Self {
        Self {
            prelude: vec![None; cfg.prelude_layers],
            recurrent: None,
            coda: vec![None; cfg.coda_layers],
            seq_len: 0,
        }
    }

    /// Positions processed so far.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Clear all cached state.
    pub fn reset(&mut self) {
        for c in &mut self.prelude {
            *c = None;
        }
        self.recurrent = None;
        for c in &mut self.coda {
            *c = None;
        }
        self.seq_len = 0;
    }
}

/// The OpenMythos recurrent-depth language model.
pub struct OpenMythos {
    embed: candle_nn::Embedding,
    prelude: Vec<TransformerBlock>,
    recurrent: RecurrentBlock,
    coda: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    head: candle_nn::Linear,
    cfg: MythosConfig,
    device: Device,
    dtype: DType,
    /// Recurrent-depth telemetry.
    pub telemetry: DepthTelemetry,
}

impl OpenMythos {
    /// Build the model from a [`VarBuilder`].
    pub fn load(vb: VarBuilder, cfg: MythosConfig) -> Result<Self> {
        cfg.validate()?;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let embed = candle_nn::embedding(cfg.vocab_size, cfg.dim, vb.pp("embed")).map_err(cand)?;

        let pre_vb = vb.pp("prelude");
        let mut prelude = Vec::with_capacity(cfg.prelude_layers);
        for i in 0..cfg.prelude_layers {
            prelude.push(TransformerBlock::load(pre_vb.pp(i), &cfg, false)?);
        }

        let recurrent = RecurrentBlock::load(vb.pp("recurrent"), &cfg)?;

        let coda_vb = vb.pp("coda");
        let mut coda = Vec::with_capacity(cfg.coda_layers);
        for i in 0..cfg.coda_layers {
            coda.push(TransformerBlock::load(coda_vb.pp(i), &cfg, false)?);
        }

        let final_norm =
            candle_nn::rms_norm(cfg.dim, cfg.rms_norm_eps, vb.pp("final_norm")).map_err(cand)?;
        let head =
            candle_nn::linear_no_bias(cfg.dim, cfg.vocab_size, vb.pp("head")).map_err(cand)?;

        Ok(Self {
            embed,
            prelude,
            recurrent,
            coda,
            final_norm,
            head,
            cfg,
            device,
            dtype,
            telemetry: DepthTelemetry::new(),
        })
    }

    /// Load from safetensors weight files.
    ///
    /// Validates `meta` against the honest boundary first (pass GGUF/HF metadata;
    /// an empty map is rejected). Tensor names must follow the module hierarchy
    /// (`embed`, `prelude.{i}.*`, `recurrent.*`, `coda.{i}.*`, `final_norm`,
    /// `head`).
    pub fn from_safetensors(
        paths: &[PathBuf],
        cfg: MythosConfig,
        meta: &BTreeMap<String, String>,
        device: &Device,
    ) -> Result<Self> {
        validate_mythos_metadata(meta)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(paths, DType::F32, device).map_err(cand)?
        };
        Self::load(vb, cfg)
    }

    /// Read GGUF metadata, enforce the honest boundary, and derive a config.
    ///
    /// Returns the validated [`MythosConfig`]. Quantized tensor loading is
    /// deferred to a real checkpoint (substrate requirement); use
    /// [`Self::from_safetensors`] for the f32 path.
    pub fn config_from_gguf(path: &std::path::Path) -> Result<MythosConfig> {
        let mut file =
            std::fs::File::open(path).map_err(|e| RuvLLMError::Gguf(format!("open gguf: {e}")))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| RuvLLMError::Gguf(format!("read gguf: {e}")))?;

        // Project GGUF metadata into a string map for validation / config.
        let mut meta = BTreeMap::new();
        for (k, v) in content.metadata.iter() {
            meta.insert(k.clone(), gguf_value_to_string(v));
        }
        validate_mythos_metadata(&meta)?;
        Ok(MythosConfig::from_metadata(&meta))
    }

    pub fn config(&self) -> &MythosConfig {
        &self.cfg
    }

    /// Stateless forward over `input_ids` `[batch, seq]` (u32) using
    /// `max_loop_iters` recurrent iterations. Returns logits `[batch, seq, vocab]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_loops(input_ids, self.cfg.max_loop_iters)
    }

    /// Stateless forward with an explicit loop count (depth extrapolation).
    pub fn forward_with_loops(&self, input_ids: &Tensor, n_loops: usize) -> Result<Tensor> {
        let mut cache = MythosCache::new(&self.cfg);
        self.forward_cached(input_ids, &mut cache, n_loops)
    }

    /// Forward that reads and updates `cache` for incremental decode. Processes
    /// the `seq` new positions in `input_ids` against the cached prefix.
    pub fn forward_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut MythosCache,
        n_loops: usize,
    ) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2().map_err(cand)?;
        let offset = cache.seq_len;
        let n_loops = n_loops.max(1);

        let mut x = self
            .embed
            .forward(input_ids)
            .map_err(cand)?
            .to_dtype(self.dtype)
            .map_err(cand)?;

        // RoPE tables for the attention variant's rotary dimension.
        let rope_dim = match self.cfg.attn_type {
            AttnType::Gqa => self.cfg.head_dim(),
            AttnType::Mla => self.cfg.qk_rope_head_dim,
        };
        let (cos, sin) = rope_tables(
            seq,
            offset,
            rope_dim,
            self.cfg.rope_theta,
            &self.device,
            self.dtype,
        )?;
        let mask = causal_mask(seq, offset + seq, offset, &self.device, self.dtype)?;

        // Prelude.
        for (i, blk) in self.prelude.iter().enumerate() {
            let past = cache.prelude[i].as_ref();
            let (out, kv) = blk.forward(&x, &cos, &sin, &mask, past)?;
            cache.prelude[i] = Some(kv);
            x = out;
        }

        // Freeze encoded input for re-injection at every loop step.
        let e = x.clone();
        let rec = self.recurrent.forward(
            &x,
            &e,
            &cos,
            &sin,
            &mask,
            cache.recurrent.as_ref(),
            n_loops,
            &self.telemetry,
        )?;
        cache.recurrent = Some(rec.kv);
        x = rec.hidden;

        // Coda.
        for (i, blk) in self.coda.iter().enumerate() {
            let past = cache.coda[i].as_ref();
            let (out, kv) = blk.forward(&x, &cos, &sin, &mask, past)?;
            cache.coda[i] = Some(kv);
            x = out;
        }

        cache.seq_len += seq;

        let x = self.final_norm.forward(&x).map_err(cand)?;
        self.head.forward(&x).map_err(cand)
    }

    /// Greedy autoregressive generation from a single-sequence prompt.
    ///
    /// Returns the newly generated token ids. Uses the KV cache for O(1)
    /// per-step attention growth; `n_loops` recurrent iterations per token (use
    /// `cfg.max_loop_iters` for the default depth). Stops early on `eos`.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        n_loops: usize,
        eos: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(RuvLLMError::Generation("empty prompt".into()));
        }
        let mut cache = MythosCache::new(&self.cfg);

        let prompt = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &self.device)
            .map_err(cand)?;
        let logits = self.forward_cached(&prompt, &mut cache, n_loops)?;
        let mut next = self.last_argmax(&logits)?;

        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            out.push(next);
            if Some(next) == eos {
                break;
            }
            let step = Tensor::from_vec(vec![next], (1, 1), &self.device).map_err(cand)?;
            let logits = self.forward_cached(&step, &mut cache, n_loops)?;
            next = self.last_argmax(&logits)?;
        }
        Ok(out)
    }

    /// Autoregressive generation with configurable sampling
    /// (temperature / top-k / top-p / repetition penalty). `n_loops` is the
    /// recurrent depth per token. Stops early on `eos`.
    pub fn generate_sampled(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        n_loops: usize,
        eos: Option<u32>,
        sampling: SamplingConfig,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(RuvLLMError::Generation("empty prompt".into()));
        }
        let mut sampler = Sampler::new(sampling);
        let mut cache = MythosCache::new(&self.cfg);
        let mut history: Vec<u32> = prompt_ids.to_vec();

        let prompt = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &self.device)
            .map_err(cand)?;
        let logits = self.forward_cached(&prompt, &mut cache, n_loops)?;
        let mut next = sampler.sample(&self.last_logits(&logits)?, &history);

        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            out.push(next);
            history.push(next);
            if Some(next) == eos {
                break;
            }
            let step = Tensor::from_vec(vec![next], (1, 1), &self.device).map_err(cand)?;
            let logits = self.forward_cached(&step, &mut cache, n_loops)?;
            next = sampler.sample(&self.last_logits(&logits)?, &history);
        }
        Ok(out)
    }

    /// Mean-pooled token embedding `[dim]` for `ids` — a lightweight sentence
    /// embedding from the input embedding layer.
    pub fn embed_pooled(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(RuvLLMError::Generation("empty input".into()));
        }
        let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device).map_err(cand)?;
        let x = self
            .embed
            .forward(&t)
            .map_err(cand)?
            .to_dtype(self.dtype)
            .map_err(cand)?;
        let pooled = x.mean(1).map_err(cand)?; // [1, dim]
        pooled
            .reshape((self.cfg.dim,))
            .map_err(cand)?
            .to_dtype(DType::F32)
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)
    }

    /// Last-position logits row `[vocab]` as host floats.
    fn last_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        let (_b, seq, _v) = logits.dims3().map_err(cand)?;
        let last = logits.i((0, seq - 1)).map_err(cand)?;
        last.to_dtype(DType::F32)
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)
    }

    /// Argmax over the vocabulary at the last sequence position of `[1, seq, vocab]`.
    fn last_argmax(&self, logits: &Tensor) -> Result<u32> {
        let (_b, seq, _v) = logits.dims3().map_err(cand)?;
        let last = logits.i((0, seq - 1)).map_err(cand)?; // [vocab]
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
}

fn gguf_value_to_string(v: &candle_core::quantized::gguf_file::Value) -> String {
    use candle_core::quantized::gguf_file::Value;
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::String(x) => x.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn model(cfg: MythosConfig) -> OpenMythos {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        OpenMythos::load(vb, cfg).expect("load")
    }

    /// Model with randomly-initialized weights (non-degenerate activations), so
    /// KV-cache parity is a real test rather than `0 == 0`.
    fn rand_model(cfg: MythosConfig) -> OpenMythos {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        OpenMythos::load(vb, cfg).expect("load")
    }

    fn meta_ok() -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("general.architecture".into(), "openmythos".into());
        m
    }

    #[test]
    fn config_validates() {
        assert!(MythosConfig::tiny().validate().is_ok());
        assert!(MythosConfig::tiny_mla().validate().is_ok());
        assert!(MythosConfig::default().validate().is_ok());
        let mut c = MythosConfig::tiny();
        c.loop_dim = 7; // odd
        assert!(c.validate().is_err());
    }

    #[test]
    fn honest_boundary() {
        assert!(validate_mythos_metadata(&meta_ok()).is_ok());
        let mut m = BTreeMap::new();
        m.insert("general.architecture".into(), "openmythos-mla".into());
        assert!(validate_mythos_metadata(&m).is_ok());
        let mut bad = BTreeMap::new();
        bad.insert("general.architecture".into(), "llama".into());
        assert!(validate_mythos_metadata(&bad).is_err());
        assert!(validate_mythos_metadata(&BTreeMap::new()).is_err());
    }

    #[test]
    fn gqa_forward_shapes() {
        let cfg = MythosConfig::tiny();
        let m = model(cfg.clone());
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (1, 5), &Device::Cpu).unwrap();
        let logits = m.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mla_forward_shapes() {
        let cfg = MythosConfig::tiny_mla();
        let m = model(cfg.clone());
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let logits = m.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn batched_forward() {
        let cfg = MythosConfig::tiny();
        let m = model(cfg.clone());
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        assert_eq!(m.forward(&ids).unwrap().dims(), &[2, 3, cfg.vocab_size]);
    }

    #[test]
    fn dense_ffn_variant_runs() {
        let mut cfg = MythosConfig::tiny();
        cfg.use_moe = false;
        let m = model(cfg.clone());
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        assert_eq!(m.forward(&ids).unwrap().dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn act_halts_via_cumulative_probability() {
        // sigmoid(0)=0.5: cumulative reaches the 0.99 threshold on step 2.
        let cfg = MythosConfig::tiny();
        let m = model(cfg);
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        let _ = m.forward(&ids).unwrap();
        let s = m.telemetry.stats();
        assert_eq!(s.max_inference_depth, 2);
        assert_eq!(s.min_inference_depth, 2);
    }

    #[test]
    fn depth_extrapolation_is_bounded() {
        let cfg = MythosConfig::tiny();
        let m = model(cfg.clone());
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        let extra = cfg.max_loop_iters + 4;
        let logits = m.forward_with_loops(&ids, extra).unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        let s = m.telemetry.stats();
        assert!(s.max_inference_depth >= 1 && s.max_inference_depth <= extra);
    }

    // ---- KV-cache decode parity ----
    //
    // At n_loops=1 the recurrent block runs a single iteration, so each token's
    // recurrent K/V is computed once from the frozen prelude output `e`; caching
    // is then provably exact and incremental decode must match a full forward
    // bit-for-bit (within fp tolerance). (For n_loops>1 the per-iteration
    // cross-token attention coupling means final-state caching is necessarily an
    // approximation, so we don't assert exact parity there.)

    #[test]
    fn cached_decode_matches_full_forward_gqa() {
        cached_matches_full(MythosConfig::tiny(), 1);
    }

    #[test]
    fn cached_decode_matches_full_forward_mla() {
        cached_matches_full(MythosConfig::tiny_mla(), 1);
    }

    fn cached_matches_full(cfg: MythosConfig, n_loops: usize) {
        let m = rand_model(cfg.clone());
        let ids = vec![3u32, 7, 1, 9, 4];

        // Full forward over the whole sequence.
        let full_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu).unwrap();
        let full = m.forward_with_loops(&full_ids, n_loops).unwrap();
        let full_last: Vec<f32> = full.i((0, ids.len() - 1)).unwrap().to_vec1().unwrap();

        // Incremental decode, one token at a time.
        let mut cache = MythosCache::new(&cfg);
        let mut last: Vec<f32> = vec![];
        for (k, &tok) in ids.iter().enumerate() {
            let step = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu).unwrap();
            let logits = m.forward_cached(&step, &mut cache, n_loops).unwrap();
            assert_eq!(cache.len(), k + 1);
            last = logits.i((0, 0)).unwrap().to_vec1().unwrap();
        }

        assert_eq!(full_last.len(), last.len());
        let max_diff = full_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "KV-cache decode diverged: max diff {max_diff}"
        );
    }

    #[test]
    fn generate_produces_tokens() {
        let cfg = MythosConfig::tiny();
        let m = model(cfg.clone());
        let out = m.generate(&[1, 2, 3], 5, cfg.max_loop_iters, None).unwrap();
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|&t| (t as usize) < cfg.vocab_size));
    }

    #[test]
    fn generate_stops_on_eos() {
        // With zero weights argmax is deterministic; force eos = that token.
        let cfg = MythosConfig::tiny();
        let m = model(cfg.clone());
        let first = m.generate(&[1, 2, 3], 1, cfg.max_loop_iters, None).unwrap()[0];
        let out = m
            .generate(&[1, 2, 3], 10, cfg.max_loop_iters, Some(first))
            .unwrap();
        assert_eq!(out.len(), 1, "should stop immediately on eos");
    }

    #[test]
    fn generate_sampled_is_in_vocab_and_deterministic_when_seeded() {
        let cfg = MythosConfig::tiny();
        let m = rand_model(cfg.clone());
        let sc = crate::models::sampling::SamplingConfig {
            temperature: 0.8,
            seed: 123,
            ..Default::default()
        };
        let a = m
            .generate_sampled(&[1, 2, 3], 6, cfg.max_loop_iters, None, sc.clone())
            .unwrap();
        let b = m
            .generate_sampled(&[1, 2, 3], 6, cfg.max_loop_iters, None, sc)
            .unwrap();
        assert_eq!(a, b, "same seed must reproduce the sequence");
        assert!(a.iter().all(|&t| (t as usize) < cfg.vocab_size));
    }

    // ---- checkpoint save / load round-trip (#2) ----

    #[test]
    fn safetensors_round_trip_preserves_logits() {
        let cfg = MythosConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let m = OpenMythos::load(vb, cfg.clone()).unwrap();

        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let before: Vec<f32> = m
            .forward(&ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        varmap.save(&path).unwrap();

        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), "openmythos".into());
        let m2 = OpenMythos::from_safetensors(&[path], cfg, &meta, &Device::Cpu).unwrap();
        let after: Vec<f32> = m2
            .forward(&ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let max_diff = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-5, "round-trip logits diverged: {max_diff}");
    }

    #[test]
    fn from_safetensors_rejects_non_mythos_metadata() {
        // Honest boundary still applies to the loader.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _ = OpenMythos::load(vb, MythosConfig::tiny()).unwrap();
        varmap.save(&path).unwrap();

        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), "llama".into());
        assert!(
            OpenMythos::from_safetensors(&[path], MythosConfig::tiny(), &meta, &Device::Cpu)
                .is_err()
        );
    }

    // ---- training: gradients flow and loss decreases (#9) ----

    #[test]
    fn train_step_reduces_loss() {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};

        // Dense FFN so every parameter on the path receives gradient.
        let mut cfg = MythosConfig::tiny();
        cfg.use_moe = false;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let m = OpenMythos::load(vb, cfg.clone()).unwrap();

        // Tiny memorization task: reproduce a fixed token sequence.
        let ids = vec![1u32, 5, 9, 13];
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(ids.clone(), (ids.len(),), &Device::Cpu).unwrap();

        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();

        let mut first = None;
        let mut last = 0f32;
        for step in 0..25 {
            let logits = m.forward(&input).unwrap();
            let logits2d = logits.reshape((ids.len(), cfg.vocab_size)).unwrap();
            let loss = candle_nn::loss::cross_entropy(&logits2d, &targets).unwrap();
            opt.backward_step(&loss).unwrap();
            let lv = loss.to_scalar::<f32>().unwrap();
            if step == 0 {
                first = Some(lv);
            }
            last = lv;
        }
        let first = first.unwrap();
        assert!(
            last < first * 0.9,
            "training did not reduce loss: {first} -> {last}"
        );
    }
}
