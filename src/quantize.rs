//! Quantize per-layer F16 shards into 4/8-bit GGUF for streaming.
//!
//! Instead of streaming F16 weights (~0.4 GB/layer for Llama-7B,
//! ~5.7 GB/layer for Mixtral-8x7B), write each layer's bulk weight
//! matrices as 4-bit or 8-bit GGUF and stream that. Reduces per-layer
//! disk I/O and device residency proportionally (~4x for Q4, ~2x for
//! Q8), at the cost of a small dequantize step per matmul.
//!
//! Conservatively keeps norms and token embeddings at F16 — the
//! quality cost of quantizing those is disproportionate to the size
//! savings (they're already small).
//!
//! This module handles the *quantizer* side (taking F16 shards and
//! producing GGUF). A quantized inference path (using `QMatMul`
//! instead of `Linear` in the streaming Block) is the natural
//! follow-up.

use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::Device;

use crate::error::{Error, Result};

/// Target quantization format. Maps to ggml dtypes; Q4_K is generally
/// preferred for quality, Q4_0 for speed, Q8_0 when 4-bit is too lossy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    Q2K,
    Q3K,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl Quantization {
    pub fn as_ggml(self) -> GgmlDType {
        match self {
            Quantization::Q2K => GgmlDType::Q2K,
            Quantization::Q3K => GgmlDType::Q3K,
            Quantization::Q4_0 => GgmlDType::Q4_0,
            Quantization::Q4K => GgmlDType::Q4K,
            Quantization::Q5K => GgmlDType::Q5K,
            Quantization::Q6K => GgmlDType::Q6K,
            Quantization::Q8_0 => GgmlDType::Q8_0,
        }
    }

    pub fn ratio_vs_f16(self) -> f32 {
        match self {
            Quantization::Q2K => 16.0 / 2.6,
            Quantization::Q3K => 16.0 / 3.4,
            Quantization::Q4_0 => 16.0 / 4.5,
            Quantization::Q4K => 16.0 / 4.5,
            Quantization::Q5K => 16.0 / 5.5,
            Quantization::Q6K => 16.0 / 6.6,
            Quantization::Q8_0 => 16.0 / 8.5,
        }
    }
}

/// True for tensors that should stay F16 even in a quantized shard.
/// Bulk weight matrices (q/k/v/o_proj, gate/up/down_proj, dense, etc.)
/// get quantized; everything else (norms, biases, embeddings, vocab
/// projections) stays full-precision. The check is name-based because
/// that's the only signal we have at quantize time.
///
/// Recognizes the naming conventions of every family ported in this
/// crate (Llama/Mistral/Qwen2/InternLM use `embed_tokens` + `lm_head`;
/// ChatGLM uses `transformer.embedding` + `transformer.output_layer`;
/// Baichuan uses Llama-compatible names + `W_pack`).
fn should_quantize(name: &str) -> bool {
    // Biases: never.
    if name.ends_with(".bias") {
        return false;
    }
    // Norms (any variant — RmsNorm, LayerNorm, post_attention_layernorm,
    // input_layernorm, model.norm, final_layernorm, etc.).
    if name.ends_with("layernorm.weight") || name.ends_with("norm.weight") {
        return false;
    }
    // Embeddings — any flavor:
    //   - Llama:      model.embed_tokens.weight
    //   - ChatGLM:    transformer.embedding.word_embeddings.weight
    //   - Baichuan:   model.embed_tokens.weight
    if name.contains("embed_tokens")
        || name.contains("word_embeddings")
        || name.contains(".embedding.")
        || name == "embedding.weight"
    {
        return false;
    }
    // Vocab projection (lm_head equivalent) — any flavor:
    //   - Llama:      lm_head.weight
    //   - ChatGLM:    transformer.output_layer.weight
    if name.starts_with("lm_head")
        || name.contains(".output_layer.")
        || name.ends_with(".output_layer.weight")
    {
        return false;
    }
    // Everything else that looks like a weight matrix.
    name.ends_with(".weight")
}

/// Quantize one per-layer safetensors shard into a per-layer GGUF.
/// `src` is e.g. `/path/splitted_model/model.layers.7.safetensors`;
/// `dst` is e.g. `/path/splitted_model_q4k/model.layers.7.gguf`.
pub fn quantize_shard(src: &Path, dst: &Path, quant: Quantization) -> Result<QuantizeReport> {
    let tensors = candle_core::safetensors::load(src, &Device::Cpu)?;
    let mut q_tensors: Vec<(String, QTensor)> = Vec::with_capacity(tensors.len());
    let mut report = QuantizeReport {
        src: src.to_path_buf(),
        dst: dst.to_path_buf(),
        total: 0,
        quantized: 0,
        kept_f16: 0,
    };

    for (name, t) in tensors {
        report.total += 1;
        let qt = if should_quantize(&name) && t.dims().len() == 2 {
            // Q4 family requires the last dim divisible by 32 (block size).
            // If a weight matrix doesn't fit, fall back to F16 rather than
            // erroring — happens for some lm_head-sized matrices.
            if t.dim(t.dims().len() - 1)? % quant_block_size(quant) == 0 {
                report.quantized += 1;
                QTensor::quantize(&t, quant.as_ggml())?
            } else {
                report.kept_f16 += 1;
                QTensor::quantize(&t, GgmlDType::F16)?
            }
        } else {
            report.kept_f16 += 1;
            QTensor::quantize(&t, GgmlDType::F16)?
        };
        q_tensors.push((name, qt));
    }

    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = File::create(dst)?;
    let metadata: [(&str, &gguf_file::Value); 0] = [];
    let tensor_refs: Vec<(&str, &QTensor)> =
        q_tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut w, &metadata, &tensor_refs)?;

    Ok(report)
}

/// Quantize every `.safetensors` shard in `src_dir` into matching
/// `.gguf` files in `dst_dir`, preserving the per-layer naming.
pub fn quantize_split_directory(
    src_dir: &Path,
    dst_dir: &Path,
    quant: Quantization,
) -> Result<Vec<QuantizeReport>> {
    let mut reports = Vec::new();
    for entry in std::fs::read_dir(src_dir)? {
        let entry = entry?;
        let src = entry.path();
        let name = match src.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let stem = match name.strip_suffix(".safetensors") {
            Some(s) => s,
            None => continue,
        };
        let dst = dst_dir.join(format!("{stem}.gguf"));
        reports.push(quantize_shard(&src, &dst, quant)?);
    }
    if reports.is_empty() {
        return Err(Error::Msg(format!(
            "no .safetensors found under {}",
            src_dir.display()
        )));
    }
    Ok(reports)
}

#[derive(Debug, Clone)]
pub struct QuantizeReport {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub total: usize,
    pub quantized: usize,
    pub kept_f16: usize,
}

/// Block size for the last-dim divisibility check. Legacy ggml quants
/// (Q4_0, Q8_0) use 32-wide blocks; k-quants (Q4_K) use 256-wide
/// superblocks. Tensors with last_dim below or not divisible by the
/// block size fall back to F16 storage.
fn quant_block_size(q: Quantization) -> usize {
    match q {
        Quantization::Q4_0 | Quantization::Q8_0 => 32,
        Quantization::Q2K
        | Quantization::Q3K
        | Quantization::Q4K
        | Quantization::Q5K
        | Quantization::Q6K => 256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn random_weight(rows: usize, cols: usize) -> Tensor {
        Tensor::randn(0f32, 1f32, (rows, cols), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    #[test]
    fn quantize_then_read_roundtrips() {
        // Use Q4_0 with 64-wide tensors so the 32-element block size
        // divides cleanly; Q4K (256-wide superblock) is covered by a
        // separate test below.
        let dir = tempdir().unwrap();
        let src = dir.path().join("model.layers.0.safetensors");
        let dst = dir.path().join("model.layers.0.gguf");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            random_weight(64, 64),
        );
        t.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            random_weight(64, 64),
        );
        t.insert(
            "model.layers.0.input_layernorm.weight".into(),
            Tensor::ones((64,), DType::F16, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&t, &src).unwrap();

        let report = quantize_shard(&src, &dst, Quantization::Q4_0).unwrap();
        assert_eq!(report.total, 3);
        assert_eq!(report.quantized, 2, "two weight matrices should quantize");
        assert_eq!(report.kept_f16, 1, "the norm stays f16");

        // Read back and verify shapes round-trip via dequantize.
        let mut f = std::fs::File::open(&dst).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        let q = content
            .tensor(
                &mut f,
                "model.layers.0.self_attn.q_proj.weight",
                &Device::Cpu,
            )
            .unwrap();
        let deq = q.dequantize(&Device::Cpu).unwrap();
        assert_eq!(deq.dims(), &[64, 64]);
    }

    #[test]
    fn q4k_requires_256_wide_tensors_and_falls_back() {
        // Q4K's 256-wide superblock means a 64-wide tensor can't be
        // Q4K-encoded — it should silently fall back to F16 instead of
        // erroring, since real models occasionally have non-aligned
        // weight matrices (e.g., some lm_head sizes).
        let dir = tempdir().unwrap();
        let src = dir.path().join("model.layers.0.safetensors");
        let dst = dir.path().join("model.layers.0.gguf");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            random_weight(64, 64),
        );
        candle_core::safetensors::save(&t, &src).unwrap();

        let report = quantize_shard(&src, &dst, Quantization::Q4K).unwrap();
        assert_eq!(report.quantized, 0);
        assert_eq!(report.kept_f16, 1);
    }

    #[test]
    fn classifier_filters_correctly() {
        // Llama-family weight matrices: quantize.
        assert!(should_quantize("model.layers.0.self_attn.q_proj.weight"));
        assert!(should_quantize("model.layers.5.mlp.gate_proj.weight"));
        // Llama-family norms / embeddings / lm_head / biases: keep F16.
        assert!(!should_quantize("model.layers.0.input_layernorm.weight"));
        assert!(!should_quantize("model.norm.weight"));
        assert!(!should_quantize("model.embed_tokens.weight"));
        assert!(!should_quantize("lm_head.weight"));
        assert!(!should_quantize("model.layers.0.self_attn.q_proj.bias"));
    }

    #[test]
    fn classifier_covers_chatglm_names() {
        // Regression: ChatGLM's embed and lm_head equivalents were
        // missing from the name filter and would get incorrectly
        // quantized. All three should stay F16.
        assert!(!should_quantize(
            "transformer.embedding.word_embeddings.weight"
        ));
        assert!(!should_quantize("transformer.output_layer.weight"));
        assert!(!should_quantize(
            "transformer.encoder.final_layernorm.weight"
        ));
        // But ChatGLM's bulk projection weights still quantize:
        assert!(should_quantize(
            "transformer.encoder.layers.5.self_attention.dense.weight"
        ));
        assert!(should_quantize(
            "transformer.encoder.layers.5.mlp.dense_h_to_4h.weight"
        ));
    }

    #[test]
    fn classifier_covers_baichuan_names() {
        // Baichuan-13B uses Llama-style embed_tokens + lm_head plus a
        // combined W_pack for QKV. The W_pack is a regular weight
        // matrix and SHOULD quantize.
        assert!(should_quantize("model.layers.0.self_attn.W_pack.weight"));
        assert!(!should_quantize("model.embed_tokens.weight"));
        assert!(!should_quantize("lm_head.weight"));
    }

    #[test]
    fn compression_ratios_sane() {
        assert!(Quantization::Q4_0.ratio_vs_f16() > 3.0);
        assert!(Quantization::Q4_0.ratio_vs_f16() < 4.0);
        assert!(Quantization::Q8_0.ratio_vs_f16() > 1.5);
        assert!(Quantization::Q8_0.ratio_vs_f16() < 2.5);
    }
}
