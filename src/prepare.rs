//! End-to-end shard preparation: download F16 weights → split per-layer →
//! quantize to GGUF. Each step is idempotent: if outputs already exist the
//! step is skipped so reruns after a crash are safe.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::quantize::Quantization;
use crate::shard::ShardConfig;

/// Options for [`ensure_q4_shards`].
pub struct PrepareOptions {
    /// Quantization format for the GGUF shards (default: Q4K).
    pub quantization: Quantization,
    /// Where to write the per-layer F16 safetensors shards produced by
    /// the split step. Defaults to `<q4_dir>/../f16_splits`.
    pub f16_splits_dir: Option<PathBuf>,
    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,
    /// Shard config (layer name prefixes). Defaults to Llama-style.
    pub shard_config: ShardConfig,
    /// How many times to retry the download on network errors (default: 10).
    pub download_retries: u32,
    /// Seconds to wait between download retries (default: 5).
    pub retry_delay_secs: u64,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            quantization: Quantization::Q4K,
            f16_splits_dir: None,
            hf_token: None,
            shard_config: ShardConfig::default(),
            download_retries: 10,
            retry_delay_secs: 5,
        }
    }
}

/// Ensure quantized GGUF shards exist for `model_id` under `q4_dir`.
///
/// Steps (each skipped if output already exists):
/// 1. Download F16 weights from HuggingFace.
/// 2. Split into per-layer safetensors shards.
/// 3. Quantize each shard to GGUF.
///
/// Returns `q4_dir` on success so callers can pass it straight to
/// [`crate::StreamingLlamaQuantized::from_pretrained`].
pub fn ensure_q4_shards(
    model_id: &str,
    q4_dir: &Path,
    opts: PrepareOptions,
) -> Result<PathBuf> {
    // Step 3 already done?
    if q4_shards_complete(q4_dir) {
        eprintln!("[prepare] Q4 shards already present at {}", q4_dir.display());
        return Ok(q4_dir.to_path_buf());
    }

    // Resolve F16 splits dir.
    let f16_dir = opts.f16_splits_dir.unwrap_or_else(|| {
        q4_dir
            .parent()
            .unwrap_or(Path::new("."))
            .join("f16_splits")
    });

    // Step 1: download with retries.
    // hf-hub/ureq doesn't support HTTP range requests, so a dropped connection
    // retries from byte 0 of the in-flight file. Already-completed blobs are
    // cached and skipped, so retrying the full call makes forward progress.
    eprintln!("[prepare] Downloading F16 weights for {model_id}...");
    let model_dir = download_with_retries(model_id, opts.hf_token.as_deref(), opts.download_retries, opts.retry_delay_secs)?;
    eprintln!("[prepare] Weights at {}", model_dir.display());

    // Steps 2+3 interleaved: per-layer shard → quantize → delete F16.
    // Keeps peak F16 footprint to a single layer (~1.6 GB) instead of
    // the full ~133 GB shard set, which matters on smaller disks.
    let splitted = f16_dir.join("splitted_model");
    std::fs::create_dir_all(&splitted)?;
    std::fs::create_dir_all(q4_dir)?;
    eprintln!(
        "[prepare] Streaming shard+quantize ({:?}) → {}",
        opts.quantization,
        q4_dir.display()
    );
    let n = shard_quantize_streaming(
        &model_dir,
        &splitted,
        q4_dir,
        opts.quantization,
        &opts.shard_config,
    )?;
    eprintln!("[prepare] Produced {n} new GGUF shards.");

    if !q4_shards_complete(q4_dir) {
        return Err(Error::Msg(format!(
            "quantization finished but shards look incomplete under {}",
            q4_dir.display()
        )));
    }

    eprintln!("[prepare] Done. Q4 shards ready at {}", q4_dir.display());
    Ok(q4_dir.to_path_buf())
}

/// Streaming shard+quantize: walk the HF weight index, materialize one
/// layer's F16 shard, quantize it to GGUF, delete the F16, then move to
/// the next. Peak F16 footprint is one layer (~1.6 GB) instead of the
/// full ~133 GB shard set.
///
/// Skips any layer whose GGUF already exists. Use this for re-runs after
/// a disk-full or partial completion.
pub fn shard_quantize_streaming(
    checkpoint_dir: &Path,
    f16_dir: &Path,
    q4_dir: &Path,
    quant: Quantization,
    shard_cfg: &ShardConfig,
) -> Result<usize> {
    use std::collections::{HashMap, HashSet};
    use std::fs;

    fs::create_dir_all(f16_dir)?;
    fs::create_dir_all(q4_dir)?;

    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(Error::Msg(format!("missing {}", index_path.display())));
    }
    let index: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Msg("invalid weight_map".into()))?;

    // Discover all layer prefixes via the same logic shard_model uses.
    let names = &shard_cfg.layer_names;
    let prefix = format!("{}.", names.layer_prefix);
    let n_layers = weight_map
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .filter_map(|k| k.strip_prefix(&prefix)?.split('.').next()?.parse::<usize>().ok())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut layers: Vec<String> = vec![names.embed.clone()];
    for i in 0..n_layers {
        layers.push(format!("{}.{}", names.layer_prefix, i));
    }
    layers.push(names.norm.clone());
    layers.push(names.lm_head.clone());

    let mut shard_files: HashMap<String, PathBuf> = HashMap::new();
    for file in weight_map.values().filter_map(|v| v.as_str()) {
        shard_files
            .entry(file.to_string())
            .or_insert_with(|| checkpoint_dir.join(file));
    }

    // Precompute which layers need each shard file. Lets us delete the
    // F16 input file once we've passed its last referencing layer —
    // critical when /tmp/<model>/ holds ~131 GB and disk is tight.
    // Enabled with RUSTYLLM_DELETE_F16_INPUTS=1; off by default so the
    // checkpoint stays usable for re-runs.
    let delete_inputs = std::env::var("RUSTYLLM_DELETE_F16_INPUTS").as_deref() == Ok("1");
    let mut last_use: HashMap<String, usize> = HashMap::new();
    if delete_inputs {
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_dot = format!("{layer}.");
            for (name, file) in weight_map.iter() {
                if name.starts_with(&layer_dot) || *name == *layer {
                    if let Some(f) = file.as_str() {
                        last_use
                            .entry(f.to_string())
                            .and_modify(|v| {
                                if layer_idx > *v {
                                    *v = layer_idx;
                                }
                            })
                            .or_insert(layer_idx);
                    }
                }
            }
        }
    }

    let mut produced = 0usize;
    for (layer_idx, layer) in layers.iter().enumerate() {
        let gguf_path = q4_dir.join(format!("{layer}.gguf"));
        if gguf_path.exists() {
            // Still need to advance progressive-delete bookkeeping so a
            // restarted run frees the same files.
            if delete_inputs {
                progressive_delete_inputs(
                    layer_idx,
                    &last_use,
                    &shard_files,
                );
            }
            continue;
        }

        let layer_dot = format!("{layer}.");
        let tensor_names: Vec<&String> = weight_map
            .keys()
            .filter(|k| k.starts_with(&layer_dot) || *k == layer)
            .collect();
        if tensor_names.is_empty() {
            tracing::warn!(layer, "no tensors matched, skipping");
            continue;
        }

        let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        let mut loaded_files: HashSet<String> = HashSet::new();
        for name in &tensor_names {
            let shard_file = weight_map
                .get(*name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Msg(format!("no shard for {name}")))?;
            if loaded_files.insert(shard_file.to_string()) {
                let path = shard_files
                    .get(shard_file)
                    .ok_or_else(|| Error::Msg(format!("missing file {shard_file}")))?;
                let loaded = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;
                for (k, t) in loaded {
                    if tensor_names.iter().any(|n| **n == k) {
                        tensors.insert(k, t);
                    }
                }
            }
        }

        let f16_out = f16_dir.join(format!("{layer}.safetensors"));
        candle_core::safetensors::save(&tensors, &f16_out)?;
        drop(tensors);

        eprintln!("[stream] {} → {}", f16_out.display(), gguf_path.display());
        crate::quantize::quantize_shard(&f16_out, &gguf_path, quant)?;
        fs::remove_file(&f16_out)?;
        produced += 1;

        // After fully consuming this layer, any input shard whose last
        // referencing layer is <= current index is no longer needed.
        if delete_inputs {
            progressive_delete_inputs(layer_idx, &last_use, &shard_files);
        }
    }
    Ok(produced)
}

/// Delete input safetensors files whose last referencing layer index is
/// <= `current_layer`. Called after each layer finishes when
/// `RUSTYLLM_DELETE_F16_INPUTS=1`.
fn progressive_delete_inputs(
    current_layer: usize,
    last_use: &std::collections::HashMap<String, usize>,
    shard_files: &std::collections::HashMap<String, PathBuf>,
) {
    for (file, &last) in last_use {
        if last == current_layer {
            if let Some(path) = shard_files.get(file) {
                if path.exists() {
                    match std::fs::remove_file(path) {
                        Ok(_) => eprintln!("[stream] freed {}", path.display()),
                        Err(e) => eprintln!("[stream] could not free {}: {e}", path.display()),
                    }
                }
            }
        }
    }
}

/// Quantize every F16 shard in `f16_dir` to GGUF under `q4_dir`, deleting
/// each F16 shard as soon as its GGUF is on disk. Lets you reclaim disk
/// space while quantizing — important when the F16 shards + raw weights
/// + GGUF outputs don't all fit at once.
///
/// Skips shards whose `.gguf` already exists (idempotent).
pub fn quantize_and_delete_f16(
    f16_dir: &Path,
    q4_dir: &Path,
    quant: Quantization,
) -> Result<usize> {
    std::fs::create_dir_all(q4_dir)?;
    let mut entries: Vec<_> = std::fs::read_dir(f16_dir)?
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut done = 0usize;
    for entry in entries {
        let src = entry.path();
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let dst = q4_dir.join(format!("{stem}.gguf"));
        if dst.exists() {
            // Already quantized — safe to delete the F16 source.
            let _ = std::fs::remove_file(&src);
            let _ = std::fs::remove_file(src.with_extension("safetensors.done"));
            continue;
        }
        eprintln!("[quantize+delete] {} → {}", src.display(), dst.display());
        crate::quantize::quantize_shard(&src, &dst, quant)?;
        std::fs::remove_file(&src)?;
        let _ = std::fs::remove_file(src.with_extension("safetensors.done"));
        done += 1;
    }
    Ok(done)
}

fn download_with_retries(
    model_id: &str,
    hf_token: Option<&str>,
    max_retries: u32,
    delay_secs: u64,
) -> Result<std::path::PathBuf> {
    let mut attempt = 0u32;
    loop {
        match crate::inference::resolve_model_dir(model_id, hf_token) {
            Ok((dir, _)) => return Ok(dir),
            Err(e) => {
                let msg = e.to_string();
                let is_network = msg.contains("body closed")
                    || msg.contains("connection")
                    || msg.contains("timed out")
                    || msg.contains("reset")
                    || msg.contains("I/O error");
                if !is_network || attempt >= max_retries {
                    return Err(e);
                }
                attempt += 1;
                eprintln!(
                    "[prepare] Download interrupted ({msg}). Retry {attempt}/{max_retries} in {delay_secs}s..."
                );
                std::thread::sleep(std::time::Duration::from_secs(delay_secs));
            }
        }
    }
}

/// True when `dir` has a contiguous set of layer GGUFs (0..max) plus
/// embed + norm + lm_head. A gap in the layer sequence means the
/// pipeline was interrupted before finishing.
fn q4_shards_complete(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut has_embed = false;
    let mut has_norm = false;
    let mut has_lm_head = false;
    let mut layer_idxs: Vec<usize> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        match &*name {
            "model.embed_tokens.gguf" => has_embed = true,
            "model.norm.gguf" => has_norm = true,
            "lm_head.gguf" => has_lm_head = true,
            n if n.starts_with("model.layers.") && n.ends_with(".gguf") => {
                if let Some(idx) = n
                    .strip_prefix("model.layers.")
                    .and_then(|r| r.strip_suffix(".gguf"))
                    .and_then(|r| r.parse::<usize>().ok())
                {
                    layer_idxs.push(idx);
                }
            }
            _ => {}
        }
    }
    if !(has_embed && has_norm && has_lm_head) || layer_idxs.is_empty() {
        return false;
    }
    layer_idxs.sort_unstable();
    let max = *layer_idxs.last().unwrap();
    // Contiguous 0..=max?
    layer_idxs.len() == max + 1
}
