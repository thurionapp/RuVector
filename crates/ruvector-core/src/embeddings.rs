//! Text Embedding Providers
//!
//! This module provides a pluggable embedding system for AgenticDB.
//!
//! ## Available Providers
//!
//! - **HashEmbedding**: Fast hash-based placeholder (default, not semantic)
//! - **OnnxEmbedding**: Real semantic embeddings using ONNX Runtime (feature: `onnx-embeddings`) ✅ RECOMMENDED
//! - **LatticeEmbedding**: Real semantic embeddings using lattice-embed, pure-Rust native inference (feature: `lattice-embeddings`)
//! - **CandleEmbedding**: Real embeddings using candle-transformers (feature: `real-embeddings`)
//! - **ApiEmbedding**: External API calls (OpenAI, Anthropic, Cohere, etc.)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ruvector_core::embeddings::{EmbeddingProvider, HashEmbedding};
//!
//! // Default: Hash-based (fast, but not semantic)
//! let hash_provider = HashEmbedding::new(384);
//! let embedding = hash_provider.embed("hello world")?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## ONNX Embeddings (Recommended for Production)
//!
//! ```rust,ignore
//! use ruvector_core::embeddings::{EmbeddingProvider, OnnxEmbedding};
//!
//! // Real semantic embeddings using all-MiniLM-L6-v2
//! let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
//! let embedding = provider.embed("hello world")?;
//! // "dog" and "cat" WILL be similar (semantic understanding!)
//! ```

use crate::error::Result;
#[cfg(any(
    feature = "real-embeddings",
    feature = "api-embeddings",
    feature = "lattice-embeddings"
))]
use crate::error::RuvectorError;
use std::sync::Arc;

/// Trait for text embedding providers
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding vector for the given text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Get the dimensionality of embeddings produced by this provider
    fn dimensions(&self) -> usize;

    /// Get a description of this provider (for logging/debugging)
    fn name(&self) -> &str;
}

/// Hash-based embedding provider (placeholder, not semantic)
///
/// ⚠️ **WARNING**: This does NOT produce semantic embeddings!
/// - "dog" and "cat" will NOT be similar
/// - "dog" and "god" WILL be similar (same characters)
///
/// Use this only for:
/// - Testing
/// - Prototyping
/// - When semantic similarity is not required
#[derive(Debug, Clone)]
pub struct HashEmbedding {
    dimensions: usize,
}

impl HashEmbedding {
    /// Create a new hash-based embedding provider
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl EmbeddingProvider for HashEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0; self.dimensions];
        let bytes = text.as_bytes();

        for (i, byte) in bytes.iter().enumerate() {
            embedding[i % self.dimensions] += (*byte as f32) / 255.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "HashEmbedding (placeholder)"
    }
}

/// Real embeddings using candle-transformers
///
/// Requires feature flag: `real-embeddings`
///
/// ⚠️ **Note**: Full candle integration is complex and model-specific.
/// For production use, we recommend:
/// 1. Using the API-based providers (simpler, always up-to-date)
/// 2. Using ONNX Runtime with pre-exported models
/// 3. Implementing your own candle wrapper for your specific model
///
/// This is a stub implementation showing the structure.
/// Users should implement `EmbeddingProvider` trait for their specific models.
#[cfg(feature = "real-embeddings")]
pub mod candle {
    use super::*;

    /// Candle-based embedding provider stub
    ///
    /// This is a placeholder. For real implementation:
    /// 1. Add candle dependencies for your specific model type
    /// 2. Implement model loading and inference
    /// 3. Handle tokenization appropriately
    ///
    /// Example structure:
    /// ```rust,ignore
    /// pub struct CandleEmbedding {
    ///     model: YourModelType,
    ///     tokenizer: Tokenizer,
    ///     device: Device,
    ///     dimensions: usize,
    /// }
    /// ```
    pub struct CandleEmbedding {
        dimensions: usize,
        model_id: String,
    }

    impl CandleEmbedding {
        /// Create a stub candle embedding provider
        ///
        /// **This is not a real implementation!**
        /// For production, implement with actual model loading.
        ///
        /// # Example
        /// ```rust,no_run
        /// # #[cfg(feature = "real-embeddings")]
        /// # {
        /// use ruvector_core::embeddings::candle::CandleEmbedding;
        ///
        /// // This returns an error - real implementation required
        /// let result = CandleEmbedding::from_pretrained(
        ///     "sentence-transformers/all-MiniLM-L6-v2",
        ///     false
        /// );
        /// assert!(result.is_err());
        /// # }
        /// ```
        pub fn from_pretrained(model_id: &str, _use_gpu: bool) -> Result<Self> {
            Err(RuvectorError::ModelLoadError(format!(
                "Candle embedding support is a stub. Please:\n\
                     1. Use ApiEmbedding for production (recommended)\n\
                     2. Or implement CandleEmbedding for model: {}\n\
                     3. See docs for ONNX Runtime integration examples",
                model_id
            )))
        }
    }

    impl EmbeddingProvider for CandleEmbedding {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Err(RuvectorError::ModelInferenceError(
                "Candle embedding not implemented - use ApiEmbedding instead".to_string(),
            ))
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn name(&self) -> &str {
            "CandleEmbedding (stub - not implemented)"
        }
    }
}

#[cfg(feature = "real-embeddings")]
pub use candle::CandleEmbedding;

/// API-based embedding provider (OpenAI, Anthropic, Cohere, etc.)
///
/// Supports any API that accepts JSON and returns embeddings in a standard format.
///
/// # Example (OpenAI)
/// ```rust,no_run
/// use ruvector_core::embeddings::{EmbeddingProvider, ApiEmbedding};
///
/// let provider = ApiEmbedding::openai("sk-...", "text-embedding-3-small");
/// let embedding = provider.embed("hello world")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "api-embeddings")]
#[derive(Clone)]
pub struct ApiEmbedding {
    api_key: String,
    endpoint: String,
    model: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "api-embeddings")]
impl ApiEmbedding {
    /// Create a new API embedding provider
    ///
    /// # Arguments
    /// * `api_key` - API key for authentication
    /// * `endpoint` - API endpoint URL
    /// * `model` - Model identifier
    /// * `dimensions` - Expected embedding dimensions
    pub fn new(api_key: String, endpoint: String, model: String, dimensions: usize) -> Self {
        Self {
            api_key,
            endpoint,
            model,
            dimensions,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create OpenAI embedding provider
    ///
    /// # Models
    /// - `text-embedding-3-small` - 1536 dimensions, $0.02/1M tokens
    /// - `text-embedding-3-large` - 3072 dimensions, $0.13/1M tokens
    /// - `text-embedding-ada-002` - 1536 dimensions (legacy)
    pub fn openai(api_key: &str, model: &str) -> Self {
        let dimensions = match model {
            "text-embedding-3-large" => 3072,
            _ => 1536, // text-embedding-3-small and ada-002
        };

        Self::new(
            api_key.to_string(),
            "https://api.openai.com/v1/embeddings".to_string(),
            model.to_string(),
            dimensions,
        )
    }

    /// Create Cohere embedding provider
    ///
    /// # Models
    /// - `embed-english-v3.0` - 1024 dimensions
    /// - `embed-multilingual-v3.0` - 1024 dimensions
    pub fn cohere(api_key: &str, model: &str) -> Self {
        Self::new(
            api_key.to_string(),
            "https://api.cohere.ai/v1/embed".to_string(),
            model.to_string(),
            1024,
        )
    }

    /// Create Voyage AI embedding provider
    ///
    /// # Models
    /// - `voyage-2` - 1024 dimensions
    /// - `voyage-large-2` - 1536 dimensions
    pub fn voyage(api_key: &str, model: &str) -> Self {
        let dimensions = if model.contains("large") { 1536 } else { 1024 };

        Self::new(
            api_key.to_string(),
            "https://api.voyageai.com/v1/embeddings".to_string(),
            model.to_string(),
            dimensions,
        )
    }
}

#[cfg(feature = "api-embeddings")]
impl EmbeddingProvider for ApiEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request_body = serde_json::json!({
            "input": text,
            "model": self.model,
        });

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| {
                RuvectorError::ModelInferenceError(format!("API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RuvectorError::ModelInferenceError(format!(
                "API returned error {}: {}",
                status, error_text
            )));
        }

        let response_json: serde_json::Value = response.json().map_err(|e| {
            RuvectorError::ModelInferenceError(format!("Failed to parse response: {}", e))
        })?;

        // Handle different API response formats
        let embedding = if let Some(data) = response_json.get("data") {
            // OpenAI format: {"data": [{"embedding": [...]}]}
            data.as_array()
                .and_then(|arr| arr.first())
                .and_then(|obj| obj.get("embedding"))
                .and_then(|emb| emb.as_array())
                .ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid OpenAI response format".to_string())
                })?
        } else if let Some(embeddings) = response_json.get("embeddings") {
            // Cohere format: {"embeddings": [[...]]}
            embeddings
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|emb| emb.as_array())
                .ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid Cohere response format".to_string())
                })?
        } else {
            return Err(RuvectorError::ModelInferenceError(
                "Unknown API response format".to_string(),
            ));
        };

        let embedding_vec: Result<Vec<f32>> = embedding
            .iter()
            .map(|v| {
                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid embedding value".to_string())
                })
            })
            .collect();

        embedding_vec
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "ApiEmbedding"
    }
}

// ============================================================================
// ONNX Embeddings (Recommended for Production)
// ============================================================================

/// ONNX-based embedding provider using ONNX Runtime
///
/// Provides **real semantic embeddings** using transformer models like all-MiniLM-L6-v2.
/// This is the **recommended** embedding provider for production use.
///
/// Requires feature flag: `onnx-embeddings`
///
/// ## Features
/// - Real semantic understanding ("dog" and "cat" ARE similar)
/// - Local inference (no API calls, works offline)
/// - Fast inference (5-50ms per embedding)
/// - Automatic model download from HuggingFace
///
/// ## Supported Models
/// - `sentence-transformers/all-MiniLM-L6-v2` (384 dims, recommended)
/// - `sentence-transformers/all-mpnet-base-v2` (768 dims)
/// - `BAAI/bge-small-en-v1.5` (384 dims)
///
/// # Example
/// ```rust,ignore
/// use ruvector_core::embeddings::{EmbeddingProvider, OnnxEmbedding};
///
/// let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
/// let embedding = provider.embed("hello world")?;
/// assert_eq!(embedding.len(), 384);
/// ```
#[cfg(feature = "onnx-embeddings")]
pub mod onnx {
    use super::*;
    use crate::error::RuvectorError;
    use ort::session::Session;
    use ort::value::{Tensor, ValueType};
    use parking_lot::RwLock;
    use std::path::PathBuf;
    use tokenizers::Tokenizer;

    /// ONNX-based embedding provider
    pub struct OnnxEmbedding {
        session: RwLock<Session>,
        tokenizer: RwLock<Tokenizer>,
        dimensions: usize,
        model_id: String,
        #[allow(dead_code)]
        max_length: usize,
    }

    impl OnnxEmbedding {
        /// Load a pre-trained embedding model from HuggingFace
        ///
        /// The model will be downloaded and cached automatically.
        ///
        /// # Arguments
        /// * `model_id` - HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        ///
        /// # Example
        /// ```rust,ignore
        /// let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
        /// ```
        pub fn from_pretrained(model_id: &str) -> Result<Self> {
            let api = hf_hub::api::sync::Api::new().map_err(|e| {
                RuvectorError::ModelLoadError(format!("Failed to create HuggingFace API: {}", e))
            })?;

            let repo = api.model(model_id.to_string());

            // Download model files
            let model_path = repo
                .get("model.onnx")
                .or_else(|_| {
                    // Try alternative path for some models
                    repo.get("onnx/model.onnx")
                })
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "Failed to download ONNX model from {}: {}. \
                     Make sure the model has an ONNX export available.",
                        model_id, e
                    ))
                })?;

            let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                RuvectorError::ModelLoadError(format!(
                    "Failed to download tokenizer from {}: {}",
                    model_id, e
                ))
            })?;

            Self::from_files(&model_path, &tokenizer_path, model_id)
        }

        /// Load from local files
        ///
        /// # Arguments
        /// * `model_path` - Path to the ONNX model file
        /// * `tokenizer_path` - Path to the tokenizer.json file
        /// * `model_id` - Model identifier for logging
        pub fn from_files(
            model_path: &PathBuf,
            tokenizer_path: &PathBuf,
            model_id: &str,
        ) -> Result<Self> {
            // Initialize ONNX Runtime (returns bool, true = first init)
            let _ = ort::init().commit();

            // Load the ONNX session
            let session = Session::builder()
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "Failed to create session builder: {}",
                        e
                    ))
                })?
                .with_intra_threads(4)
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!("Failed to set thread count: {}", e))
                })?
                .commit_from_file(model_path)
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!("Failed to load ONNX model: {}", e))
                })?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
                RuvectorError::ModelLoadError(format!("Failed to load tokenizer: {}", e))
            })?;

            // Determine dimensions from model output
            let dimensions = Self::infer_dimensions(&session, model_id)?;

            // Determine max_length from model (default to 512 for sentence transformers)
            let max_length = 512;

            tracing::info!(
                "Loaded ONNX embedding model: {} ({}D)",
                model_id,
                dimensions
            );

            Ok(Self {
                session: RwLock::new(session),
                tokenizer: RwLock::new(tokenizer),
                dimensions,
                model_id: model_id.to_string(),
                max_length,
            })
        }

        fn infer_dimensions(session: &Session, model_id: &str) -> Result<usize> {
            // Common dimensions for known models
            let dimensions = match model_id {
                id if id.contains("all-MiniLM-L6") => 384,
                id if id.contains("all-mpnet-base") => 768,
                id if id.contains("bge-small") => 384,
                id if id.contains("bge-base") => 768,
                id if id.contains("bge-large") => 1024,
                id if id.contains("e5-small") => 384,
                id if id.contains("e5-base") => 768,
                id if id.contains("e5-large") => 1024,
                _ => {
                    // Try to infer from output shape via session.outputs() method
                    if let Some(output) = session.outputs().first() {
                        if let ValueType::Tensor { shape, .. } = output.dtype() {
                            let dims: Vec<i64> = shape.iter().copied().collect();
                            if dims.len() >= 2 {
                                let last_dim = dims[dims.len() - 1];
                                if last_dim > 0 {
                                    return Ok(last_dim as usize);
                                }
                            }
                        }
                    }
                    // Default to 384 (most common)
                    384
                }
            };

            Ok(dimensions)
        }

        /// Embed multiple texts in a batch (more efficient than individual calls)
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn mean_pooling(
            token_embeddings: &[f32],
            attention_mask: &[i64],
            seq_len: usize,
            hidden_size: usize,
        ) -> Vec<f32> {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut mask_sum = 0.0f32;

            for i in 0..seq_len {
                let mask = attention_mask[i] as f32;
                mask_sum += mask;
                for j in 0..hidden_size {
                    pooled[j] += token_embeddings[i * hidden_size + j] * mask;
                }
            }

            // Avoid division by zero
            if mask_sum > 0.0 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            pooled
        }
    }

    impl EmbeddingProvider for OnnxEmbedding {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Tokenize
            let encoding = {
                let tokenizer = self.tokenizer.read();
                tokenizer.encode(text, true).map_err(|e| {
                    RuvectorError::ModelInferenceError(format!("Tokenization failed: {}", e))
                })?
            };

            // Prepare inputs
            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let seq_len = input_ids.len();

            // Create ONNX tensors using ort 2.0 API (batch_size=1)
            // Tensor::from_array takes (shape, owned_data)
            let input_ids_tensor =
                Tensor::<i64>::from_array(([1, seq_len], input_ids.clone().into_boxed_slice()))
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!(
                            "Failed to create input_ids tensor: {}",
                            e
                        ))
                    })?;

            let attention_mask_tensor = Tensor::<i64>::from_array((
                [1, seq_len],
                attention_mask.clone().into_boxed_slice(),
            ))
            .map_err(|e| {
                RuvectorError::ModelInferenceError(format!(
                    "Failed to create attention_mask tensor: {}",
                    e
                ))
            })?;

            let token_type_ids_tensor =
                Tensor::<i64>::from_array(([1, seq_len], token_type_ids.into_boxed_slice()))
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!(
                            "Failed to create token_type_ids tensor: {}",
                            e
                        ))
                    })?;

            // Run inference and extract output (needs mutable access to session)
            // We must extract all data while holding the lock since SessionOutputs has a lifetime
            let (output_data, output_shape_vec) = {
                let mut session = self.session.write();
                let outputs = session
                    .run(ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor,
                        "token_type_ids" => token_type_ids_tensor,
                    ])
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!("ONNX inference failed: {}", e))
                    })?;

                // Extract output using indexing (ort 2.0 API)
                // Sentence transformers output shape: [batch_size, seq_len, hidden_size]
                let output_value = &outputs[0];

                // Extract as ndarray view
                let output_array = output_value.try_extract_array::<f32>().map_err(|e| {
                    RuvectorError::ModelInferenceError(format!(
                        "Failed to extract output tensor: {}",
                        e
                    ))
                })?;

                let output_shape_vec: Vec<usize> = output_array.shape().to_vec();
                let output_data_vec: Vec<f32> = output_array.iter().copied().collect();

                (output_data_vec, output_shape_vec)
            };

            // Determine if we need pooling based on output shape
            let embedding = if output_shape_vec.len() == 3 {
                // Shape: [batch_size, seq_len, hidden_size] - needs pooling
                let hidden_size = output_shape_vec[2];
                Self::mean_pooling(&output_data, &attention_mask, seq_len, hidden_size)
            } else if output_shape_vec.len() == 2 {
                // Shape: [batch_size, hidden_size] - already pooled
                let mut emb = output_data;
                // L2 normalize
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut emb {
                        *val /= norm;
                    }
                }
                emb
            } else {
                return Err(RuvectorError::ModelInferenceError(format!(
                    "Unexpected output shape: {:?}",
                    output_shape_vec
                )));
            };

            Ok(embedding)
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn name(&self) -> &str {
            &self.model_id
        }
    }

    impl std::fmt::Debug for OnnxEmbedding {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxEmbedding")
                .field("model_id", &self.model_id)
                .field("dimensions", &self.dimensions)
                .field("max_length", &self.max_length)
                .finish()
        }
    }
}

#[cfg(feature = "onnx-embeddings")]
pub use onnx::OnnxEmbedding;

// ============================================================================
// Lattice Embeddings (pure-Rust, native, no C++ FFI / no ONNX Runtime)
// ============================================================================

/// Native embedding provider backed by [`lattice-embed`](https://crates.io/crates/lattice-embed),
/// a pure-Rust transformer inference engine (SIMD matmul, safetensors weight
/// loading, no ONNX Runtime, no C++ FFI).
///
/// Requires feature flag: `lattice-embeddings`
///
/// ## Supported models
/// - `bge-small-en-v1.5` / `BAAI/bge-small-en-v1.5` (384 dims, default, recommended for `.rvf` packs)
/// - `bge-base-en-v1.5` / `BAAI/bge-base-en-v1.5` (768 dims)
/// - `bge-large-en-v1.5` / `BAAI/bge-large-en-v1.5` (1024 dims)
/// - `multilingual-e5-small` / `intfloat/multilingual-e5-small` (384 dims)
/// - `multilingual-e5-base` / `intfloat/multilingual-e5-base` (768 dims)
/// - `all-minilm-l6-v2` / `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
/// - `paraphrase-multilingual-minilm-l12-v2` / `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims)
/// - `qwen3-embedding-0.6b` / `Qwen/Qwen3-Embedding-0.6B` (1024 dims)
/// - `qwen3-embedding-4b` / `Qwen/Qwen3-Embedding-4B` (2560 dims)
///
/// Model-id parsing is delegated to `lattice_embed::EmbeddingModel`'s own
/// `FromStr` impl (case-insensitive, accepts display names, short names, and
/// HuggingFace ids) rather than re-implementing the mapping here, so this
/// provider stays in sync with lattice-embed's canonical model table.
///
/// ## CPU / native, no GPU
/// This provider uses lattice-embed's default `native` feature (CPU-only,
/// SIMD-accelerated). It does **not** enable lattice-embed's `metal-gpu`
/// feature.
///
/// ## Minimum Supported Rust Version
/// Enabling the `lattice-embeddings` feature raises the effective MSRV for
/// this crate to Rust 1.93 (edition 2024), since `lattice-embed` requires it.
/// Cargo has no mechanism to express a per-feature `rust-version`, so this is
/// not reflected in `rust-version.workspace = true` above — it only applies
/// when this feature is enabled. The crate's default build (feature
/// disabled) keeps the workspace MSRV of 1.77.
///
/// ## Model download
/// BERT-family models (BGE, E5, MiniLM) download automatically from
/// HuggingFace into `~/.lattice/models` on first use. Qwen3-Embedding models
/// must be placed at `~/.lattice/models/qwen3-embedding-{0.6b,4b}/` manually
/// (or pointed to via `LATTICE_QWEN_MODEL_DIR`) before first use.
///
/// ## Asymmetric retrieval (query vs. passage prefixing)
/// BGE, E5, and Qwen3-Embedding are asymmetric retrievers: the query side is
/// prefixed with a retrieval instruction, the document side is not.
/// [`EmbeddingProvider::embed`] always takes the **passage/document** side (no
/// query instruction) via `lattice_embed::EmbeddingService::embed_passage`.
/// Use the inherent [`LatticeEmbedding::embed_query`] method for query text —
/// it applies the model's query instruction via
/// `EmbeddingService::embed_query`: BGE v1.5 prefixes queries with
/// `"Represent this sentence for searching relevant passages: "`, E5 with
/// `"query: "`, and Qwen3-Embedding with its search instruction. For all three
/// families `embed_query` and `embed` therefore produce different vectors,
/// which is what makes asymmetric retrieval correct. MiniLM is genuinely
/// symmetric (contrastive training on raw text, no prefix), so its two methods
/// are equivalent.
///
/// ## Normalization
/// Both [`EmbeddingProvider::embed`] and [`LatticeEmbedding::embed_query`]
/// return L2-normalized vectors (unit length): `lattice-embed`'s BERT-family
/// encode path (used for BGE, E5, and MiniLM) calls `l2_normalize`
/// unconditionally on the pooled output, both for single-text and batched
/// encoding (`BertModel::encode` / `encode_batch` in
/// `crates/inference/src/model/bert.rs`, upstream in
/// [`lattice-embed`](https://crates.io/crates/lattice-embed)'s
/// `lattice-inference` dependency). This holds regardless of distance
/// metric — safe to use with a dot-product index as well as cosine.
///
/// # Example
/// ```rust,no_run
/// use ruvector_core::embeddings::{EmbeddingProvider, LatticeEmbedding};
///
/// let provider = LatticeEmbedding::from_pretrained("bge-small-en-v1.5")?;
///
/// // Document side: no query instruction.
/// let doc_embedding = provider.embed("The cat sat on the mat.")?;
/// assert_eq!(doc_embedding.len(), 384);
///
/// // Query side: applies the model's query instruction, if any.
/// let query_embedding = provider.embed_query("Where did the cat sit?")?;
/// assert_eq!(query_embedding.len(), 384);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "lattice-embeddings")]
pub mod lattice_native {
    use super::*;
    use lattice_embed::{
        EmbeddingModel as LatticeEmbeddingModel, EmbeddingService, NativeEmbeddingService,
    };
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::thread;

    /// Which side of asymmetric retrieval a queued embedding request is for.
    enum EmbedKind {
        Query,
        Passage,
    }

    /// A single embedding request sent to the worker thread, with a
    /// per-request reply channel for the result.
    struct EmbedRequest {
        kind: EmbedKind,
        text: String,
        reply_tx: mpsc::Sender<std::result::Result<Vec<f32>, String>>,
    }

    /// See the [module-level docs](self) for the full provider description.
    ///
    /// # Examples
    /// Embed a passage and a query on an asymmetric BGE model. The query is
    /// embedded with [`embed_query`](LatticeEmbedding::embed_query), which
    /// applies BGE's retrieval instruction, so it produces a different vector
    /// than passing the same text through [`EmbeddingProvider::embed`] (the
    /// passage side). Using `embed_query` for queries is what makes
    /// query-to-passage retrieval scores correct on asymmetric models.
    /// ```rust,no_run
    /// use ruvector_core::embeddings::{EmbeddingProvider, LatticeEmbedding};
    ///
    /// let provider = LatticeEmbedding::from_pretrained("bge-small-en-v1.5")?;
    ///
    /// let passage = provider.embed("The Eiffel Tower is in Paris, France.")?;
    /// let query = provider.embed_query("Where is the Eiffel Tower?")?;
    /// assert_eq!(passage.len(), provider.dimensions());
    /// assert_eq!(query.len(), provider.dimensions());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// A runnable version that prints the cosine similarities of the query and
    /// passage vectors is in `examples/lattice_embedding_example.rs`.
    ///
    /// # Threading model
    /// `lattice-embed`'s [`EmbeddingService`] is `async`-only (no sync/blocking
    /// API), but [`EmbeddingProvider::embed`] is a sync method that ruvector-core
    /// callers may invoke from anywhere, including from inside an existing Tokio
    /// runtime (e.g. an async server handler). Bridging via a stored
    /// `Runtime::block_on` would panic in that case (`block_on` cannot be
    /// called from within an already-running runtime). Instead, the runtime and
    /// the embedding service live on a dedicated worker thread with no ambient
    /// async context of its own; `embed` / `embed_query` send a request over a
    /// channel and block on `Receiver::recv`, which is safe to call from any
    /// context, sync or async.
    pub struct LatticeEmbedding {
        model: LatticeEmbeddingModel,
        model_id: &'static str,
        dimensions: usize,
        request_tx: Mutex<mpsc::Sender<EmbedRequest>>,
        // Keeps the worker thread's handle alive for the lifetime of this
        // provider. Not joined on drop (that would block); dropping
        // `request_tx` closes the channel, which ends the worker's `recv`
        // loop and lets the thread exit on its own.
        _worker: thread::JoinHandle<()>,
    }

    impl LatticeEmbedding {
        /// Load a pre-trained embedding model by id.
        ///
        /// Accepts display names (`"bge-small-en-v1.5"`), short names
        /// (`"bge-small"`, `"small"`), and HuggingFace ids
        /// (`"BAAI/bge-small-en-v1.5"`) — see [`lattice_embed::EmbeddingModel`]'s
        /// `FromStr` impl for the full accepted set. Returns an error for any
        /// unrecognized id, and for any id that resolves to a model
        /// [`lattice_embed`]'s native service cannot run locally (e.g. the
        /// remote-only OpenAI variants).
        ///
        /// # Example
        /// ```rust,no_run
        /// use ruvector_core::embeddings::LatticeEmbedding;
        ///
        /// let provider = LatticeEmbedding::from_pretrained("bge-small-en-v1.5")?;
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn from_pretrained(model_id: &str) -> Result<Self> {
            let model: LatticeEmbeddingModel = model_id.parse().map_err(|e: String| {
                RuvectorError::ModelLoadError(format!(
                    "unknown lattice-embed model id '{model_id}': {e}"
                ))
            })?;
            Self::with_model(model)
        }

        /// Load a pre-trained embedding model from an already-resolved
        /// [`lattice_embed::EmbeddingModel`] variant.
        pub fn with_model(model: LatticeEmbeddingModel) -> Result<Self> {
            if !model.is_local() {
                return Err(RuvectorError::ModelLoadError(format!(
                    "'{model}' cannot be loaded natively: lattice-embed's \
                     NativeEmbeddingService only supports models it can run \
                     on-device. Remote/API-only models (e.g. the OpenAI \
                     text-embedding-* family) are not supported by LatticeEmbedding."
                )));
            }

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "failed to build tokio runtime for LatticeEmbedding: {e}"
                    ))
                })?;
            let service = NativeEmbeddingService::with_model(model);

            let (request_tx, request_rx) = mpsc::channel::<EmbedRequest>();
            let worker = thread::Builder::new()
                .name("lattice-embed-worker".to_string())
                .spawn(move || {
                    // No ambient Tokio runtime exists on this thread, so
                    // `block_on` here can never panic on nested-runtime
                    // grounds regardless of the caller's own context.
                    for request in request_rx {
                        let outcome = runtime.block_on(async {
                            match request.kind {
                                EmbedKind::Query => {
                                    service.embed_query(&[request.text], model).await
                                }
                                EmbedKind::Passage => {
                                    service.embed_passage(&[request.text], model).await
                                }
                            }
                        });
                        let mapped =
                            outcome
                                .map_err(|e| e.to_string())
                                .and_then(|mut embeddings| {
                                    embeddings.pop().ok_or_else(|| {
                                        "lattice-embed returned no embedding".to_string()
                                    })
                                });
                        // Ignore send errors: they only occur if the caller
                        // already dropped its reply receiver.
                        let _ = request.reply_tx.send(mapped);
                    }
                })
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "failed to spawn LatticeEmbedding worker thread: {e}"
                    ))
                })?;

            Ok(Self {
                model,
                model_id: model.model_id(),
                dimensions: model.dimensions(),
                request_tx: Mutex::new(request_tx),
                _worker: worker,
            })
        }

        /// Get the dimensionality of embeddings produced by the loaded model.
        pub fn dimensions(&self) -> usize {
            self.dimensions
        }

        /// Embed **query** text, applying the model's query-side prompt
        /// instruction if it uses one (BGE v1.5's `"Represent this sentence
        /// for searching relevant passages: "` prefix, E5's `"query: "`
        /// prefix, Qwen3's search-query instruction). For those asymmetric
        /// models this produces a different vector than
        /// [`EmbeddingProvider::embed`]; only MiniLM is symmetric, so its two
        /// methods are equivalent.
        ///
        /// This is what makes asymmetric retrieval correct: index documents
        /// via [`EmbeddingProvider::embed`] (passage side, no prefix) and
        /// embed the search query via this method (query side, prefixed).
        ///
        /// Safe to call from any context, including from inside a Tokio
        /// runtime — see the [threading model](LatticeEmbedding#threading-model).
        pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.send_request(EmbedKind::Query, text)
        }

        /// Send an embedding request to the worker thread and block on the
        /// reply. Never calls `block_on` on the caller's thread, so this is
        /// safe to invoke from inside an existing async runtime.
        fn send_request(&self, kind: EmbedKind, text: &str) -> Result<Vec<f32>> {
            let (reply_tx, reply_rx) = mpsc::channel();
            let request = EmbedRequest {
                kind,
                text: text.to_string(),
                reply_tx,
            };

            self.request_tx
                .lock()
                .map_err(|_| {
                    RuvectorError::ModelInferenceError(
                        "lattice-embed embedding worker request channel poisoned".to_string(),
                    )
                })?
                .send(request)
                .map_err(|_| {
                    RuvectorError::ModelInferenceError(
                        "lattice-embed embedding worker unavailable".to_string(),
                    )
                })?;

            reply_rx
                .recv()
                .map_err(|_| {
                    RuvectorError::ModelInferenceError(
                        "lattice-embed embedding worker unavailable".to_string(),
                    )
                })?
                .map_err(|e| {
                    RuvectorError::ModelInferenceError(format!(
                        "lattice-embed embedding failed: {e}"
                    ))
                })
        }
    }

    impl EmbeddingProvider for LatticeEmbedding {
        /// Embed **passage/document** text (no query instruction applied).
        ///
        /// Use [`LatticeEmbedding::embed_query`] for the query side of
        /// asymmetric retrieval. Safe to call from any context, including
        /// from inside a Tokio runtime — see the
        /// [threading model](LatticeEmbedding#threading-model).
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.send_request(EmbedKind::Passage, text)
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn name(&self) -> &str {
            self.model_id
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn from_pretrained_rejects_remote_only_models() {
            // "text-embedding-3-small" and "openai" both parse successfully
            // to `EmbeddingModel::TextEmbedding3Small` (see lattice-embed's
            // `FromStr` impl) but that variant is remote/API-only —
            // `NativeEmbeddingService` cannot run it. Both aliases must be
            // rejected at construction time, not on first `embed()` call.
            assert!(
                LatticeEmbedding::from_pretrained("text-embedding-3-small").is_err(),
                "remote-only model 'text-embedding-3-small' must be rejected at construction"
            );
            assert!(
                LatticeEmbedding::from_pretrained("openai").is_err(),
                "remote-only model alias 'openai' must be rejected at construction"
            );
        }

        #[test]
        fn from_pretrained_accepts_native_local_model() {
            assert!(
                LatticeEmbedding::from_pretrained("bge-small-en-v1.5").is_ok(),
                "native local model 'bge-small-en-v1.5' must construct successfully"
            );
        }

        /// #662: pins the bge-small alias surface this provider accepts.
        /// `ruvector-extensions`' `LatticeWasmEmbeddings` (the WASM sibling of
        /// this provider) mirrors this same alias set
        /// (`normalizeLatticeWasmModel` in
        /// `npm/packages/ruvector-extensions/src/embeddings.ts`) so a model id
        /// valid for one Lattice-backed provider is valid for the other.
        #[test]
        fn from_pretrained_accepts_bge_small_alias_surface() {
            for alias in [
                "bge-small-en-v1.5",
                "bge-small-en",
                "bge-small",
                "small",
                "BAAI/bge-small-en-v1.5",
                "BGE_SMALL_EN_V1.5",
            ] {
                let provider = LatticeEmbedding::from_pretrained(alias)
                    .unwrap_or_else(|e| panic!("alias '{alias}' must resolve to bge-small: {e}"));
                assert_eq!(
                    provider.dimensions(),
                    384,
                    "alias '{alias}' resolved to the wrong dimensionality"
                );
                assert_eq!(
                    provider.name(),
                    "BAAI/bge-small-en-v1.5",
                    "alias '{alias}' resolved to a different model than 'bge-small-en-v1.5'"
                );
            }
        }

        /// Regression test for the nested-runtime panic: `embed` / `embed_query`
        /// used to call `Runtime::block_on` on a `Runtime` stored on the
        /// provider, which panics when invoked from inside an already-running
        /// Tokio runtime. The worker-thread bridge has no ambient runtime on
        /// the calling side, so both calls must succeed here instead.
        #[tokio::test]
        async fn embed_from_inside_async_runtime_does_not_panic() {
            let provider = LatticeEmbedding::from_pretrained("bge-small-en-v1.5")
                .expect("bge-small-en-v1.5 is a native local model");

            let doc = provider
                .embed("a nested-runtime regression test")
                .expect("embed must not panic or error from inside a Tokio runtime");
            assert_eq!(doc.len(), provider.dimensions());

            let query = provider
                .embed_query("a nested-runtime regression test")
                .expect("embed_query must not panic or error from inside a Tokio runtime");
            assert_eq!(query.len(), provider.dimensions());
        }

        /// Cross-provider contract test (maintainer follow-up on #663).
        ///
        /// This provider never builds the prefixed query string itself: `embed_query`
        /// forwards raw `text` to `EmbeddingService::embed_query`, which prepends
        /// `model.query_instruction()` internally (see `send_request` above and
        /// `lattice_embed::EmbeddingService::embed_query`'s default impl). So the
        /// prefix this provider *effectively* applies for a given model **is**
        /// `LatticeEmbeddingModel::query_instruction()` / `document_instruction()` --
        /// both documented `**Stable**` in lattice-embed's own API-stability
        /// convention (`crates/embed/src/model.rs` in ohdearquant/lattice).
        ///
        /// `ruvector-extensions`' WASM sibling provider has no such delegation
        /// (`@khive-ai/lattice-embed-wasm`'s `embed()` binding takes raw text only,
        /// no prefix concept), so it hardcodes the same prefixes as a TS literal
        /// map (`LATTICE_WASM_QUERY_INSTRUCTIONS` in
        /// `npm/packages/ruvector-extensions/src/embeddings.ts`) and asserts against
        /// the identical fixture in its own contract test
        /// (`npm/packages/ruvector-extensions/tests/lattice-prefix-contract.test.ts`).
        /// Both tests read `fixtures/lattice-embed/query-prefixes.json` at the repo
        /// root, so a future lattice-embed bump that changes either model's
        /// convention fails this test on the Rust side (and its TS sibling
        /// independently), instead of the two providers silently re-diverging the
        /// way they did before #663.
        #[test]
        fn cross_provider_query_prefix_contract() {
            let fixture: serde_json::Value = serde_json::from_str(include_str!(
                "../../../fixtures/lattice-embed/query-prefixes.json"
            ))
            .expect("fixtures/lattice-embed/query-prefixes.json must be valid JSON");

            let models = fixture["models"]
                .as_object()
                .expect("fixture must have a top-level 'models' object");
            assert!(
                !models.is_empty(),
                "fixture 'models' must not be empty -- an empty fixture would make this \
                 contract test vacuously pass"
            );
            assert!(
                models.contains_key("bge-small"),
                "fixture must cover 'bge-small' -- the model #662 was about"
            );
            assert!(
                models.contains_key("minilm"),
                "fixture must cover 'minilm' as the symmetric control case"
            );

            for (alias, expected) in models {
                let model: LatticeEmbeddingModel = alias.parse().unwrap_or_else(|e| {
                    panic!(
                        "fixture alias '{alias}' must be a valid lattice_embed::EmbeddingModel: {e}"
                    )
                });

                let expected_query_prefix = expected["query_prefix"].as_str();
                assert_eq!(
                    model.query_instruction(),
                    expected_query_prefix,
                    "query prefix mismatch for '{alias}': lattice_embed::EmbeddingModel::\
                     query_instruction() returned {:?} but fixtures/lattice-embed/\
                     query-prefixes.json expects {:?}. If lattice-embed intentionally changed \
                     this model's convention, update the fixture AND the TS sibling test in \
                     npm/packages/ruvector-extensions/tests/lattice-prefix-contract.test.ts \
                     together.",
                    model.query_instruction(),
                    expected_query_prefix
                );

                let expected_passage_prefix = expected["passage_prefix"].as_str();
                assert_eq!(
                    model.document_instruction(),
                    expected_passage_prefix,
                    "passage prefix mismatch for '{alias}': lattice_embed::EmbeddingModel::\
                     document_instruction() returned {:?} but the fixture expects {:?}",
                    model.document_instruction(),
                    expected_passage_prefix
                );
            }
        }
    }
}

#[cfg(feature = "lattice-embeddings")]
pub use lattice_native::LatticeEmbedding;

/// Type-erased embedding provider for dynamic dispatch
pub type BoxedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedding() {
        let provider = HashEmbedding::new(128);

        let emb1 = provider.embed("hello world").unwrap();
        let emb2 = provider.embed("hello world").unwrap();

        assert_eq!(emb1.len(), 128);
        assert_eq!(emb1, emb2, "Same text should produce same embedding");

        // Check normalization
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_hash_embedding_different_text() {
        let provider = HashEmbedding::new(128);

        let emb1 = provider.embed("hello").unwrap();
        let emb2 = provider.embed("world").unwrap();

        assert_ne!(
            emb1, emb2,
            "Different text should produce different embeddings"
        );
    }

    #[cfg(feature = "real-embeddings")]
    #[test]
    #[ignore] // Requires model download
    fn test_candle_embedding() {
        let provider =
            CandleEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", false)
                .unwrap();

        let embedding = provider.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    #[ignore] // Requires API key
    fn test_api_embedding_openai() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let provider = ApiEmbedding::openai(&api_key, "text-embedding-3-small");

        let embedding = provider.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 1536);
    }

    #[cfg(feature = "onnx-embeddings")]
    mod onnx_tests {
        use super::*;

        #[test]
        #[ignore] // Requires model download (~90MB)
        fn test_onnx_embedding_minilm() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let embedding = provider.embed("hello world").unwrap();
            assert_eq!(embedding.len(), 384);

            // Check normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Embedding should be normalized, got norm={}",
                norm
            );
        }

        #[test]
        #[ignore] // Requires model download
        fn test_onnx_semantic_similarity() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let emb_dog = provider.embed("dog").unwrap();
            let emb_cat = provider.embed("cat").unwrap();
            let emb_car = provider.embed("car").unwrap();

            // Cosine similarity (embeddings are normalized, so dot product = cosine)
            let sim_dog_cat: f32 = emb_dog.iter().zip(&emb_cat).map(|(a, b)| a * b).sum();
            let sim_dog_car: f32 = emb_dog.iter().zip(&emb_car).map(|(a, b)| a * b).sum();

            // dog and cat should be more similar than dog and car
            assert!(
                sim_dog_cat > sim_dog_car,
                "Expected dog-cat similarity ({}) > dog-car similarity ({})",
                sim_dog_cat,
                sim_dog_car
            );
        }

        #[test]
        #[ignore] // Requires model download
        fn test_onnx_batch_embedding() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let texts = vec!["hello world", "goodbye world", "rust programming"];
            let embeddings = provider.embed_batch(&texts).unwrap();

            assert_eq!(embeddings.len(), 3);
            for emb in &embeddings {
                assert_eq!(emb.len(), 384);
            }
        }
    }

    #[cfg(feature = "lattice-embeddings")]
    mod lattice_tests {
        use super::*;
        use crate::embeddings::LatticeEmbedding;

        /// Pure model-id mapping test — no network, no model load.
        /// `LatticeEmbedding::from_pretrained` delegates to
        /// `lattice_embed::EmbeddingModel::from_str`; this test locks in that
        /// bge-small resolves from both its display name and its HuggingFace
        /// id, and that an unrecognized id errors instead of silently
        /// defaulting.
        #[test]
        fn test_lattice_from_pretrained_model_id_mapping() {
            let by_display_name = LatticeEmbedding::from_pretrained("bge-small-en-v1.5").unwrap();
            assert_eq!(by_display_name.dimensions(), 384);
            assert_eq!(EmbeddingProvider::dimensions(&by_display_name), 384);

            let by_hf_id = LatticeEmbedding::from_pretrained("BAAI/bge-small-en-v1.5").unwrap();
            assert_eq!(by_hf_id.dimensions(), 384);

            let unknown = LatticeEmbedding::from_pretrained("not-a-real-model");
            assert!(
                unknown.is_err(),
                "unknown model id should error, not default"
            );
        }

        #[test]
        fn test_lattice_from_pretrained_minilm_mapping() {
            let by_short = LatticeEmbedding::from_pretrained("all-minilm-l6-v2").unwrap();
            assert_eq!(by_short.dimensions(), 384);

            let by_hf_id =
                LatticeEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                    .unwrap();
            assert_eq!(by_hf_id.dimensions(), 384);
        }

        /// Real end-to-end embedding test. Requires the bge-small-en-v1.5
        /// model to be downloaded from HuggingFace on first use (~130MB) —
        /// network access, not run in CI. Run manually with:
        ///   cargo test -p ruvector-core --features lattice-embeddings -- --ignored lattice_tests
        #[test]
        #[ignore]
        fn test_lattice_embedding_real() {
            let provider = LatticeEmbedding::from_pretrained("bge-small-en-v1.5").unwrap();

            let embedding = provider.embed("hello world").unwrap();
            assert_eq!(embedding.len(), 384);
            assert!(embedding.iter().all(|v| v.is_finite()));

            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embedding should be L2-normalized, got norm={norm}"
            );
        }
    }
}
