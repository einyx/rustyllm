//! RustyLLM — layer-wise LLM inference for constrained GPU memory.
//!
//! Loads one transformer layer at a time so large models can run on
//! small GPUs.

#[cfg(feature = "cuda")]
pub mod async_upload;
pub mod error;
pub mod family;
pub mod inference;
pub mod parallel_linear;
pub mod prepare;
pub mod q4_kernel;
pub mod quantize;
pub mod shard;
pub mod streaming;
pub mod streaming_baichuan;
pub mod streaming_chatglm;
pub mod streaming_family;
pub mod streaming_mixtral;
pub mod streaming_quantized;

pub use error::{Error, Result};
pub use family::Family;
pub use inference::LayeredLlama;
pub use prepare::{ensure_q4_shards, quantize_and_delete_f16, PrepareOptions};
pub use quantize::{
    quantize_shard, quantize_split_directory, requantize_shard_mixed, Quantization,
    QuantizeReport, TensorRole,
};
pub use shard::{shard_model, ShardConfig, ShardPaths};
pub use streaming::{StreamCache, StreamingLlama};
pub use streaming_family::StreamingFamily;
pub use streaming_quantized::{QuantStreamCache, StreamingLlamaQuantized};
