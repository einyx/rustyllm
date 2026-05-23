//! Async H2D weight loader for the streaming Block path.
//!
//! Replaces the synchronous `VarBuilder::from_mmaped_safetensors` +
//! `Block::load` pattern with a double-buffered uploader that runs
//! H2D on a separate CUDA stream, overlapping the next layer's
//! transfer with the current layer's compute.
//!
//! Architecture (per design_async_h2d.md, phases 1-3a validated as
//! standalone spikes in `examples/async_h2d_*.rs`):
//!
//! ```text
//!   layer N file ─┐
//!                 │ mmap → pinned[next] (host-host memcpy)
//!                 ▼
//!         ┌──────────────────────┐
//!         │ pinned[0] | pinned[1] │  (one being filled, one being DMA'd)
//!         └─────────┬────────────┘
//!                   │ cudaMemcpyHtoDAsync(upload_stream)
//!                   ▼
//!         ┌──────────────────────┐
//!         │ device[0] | device[1] │  (ping-pong, one current, one prefetching)
//!         └─────────┬────────────┘
//!                   │ wrap_cuda_slice + Tensor::from_storage
//!                   ▼
//!              VarBuilder ───→ Block::load on compute_stream
//! ```
//!
//! Cross-stream sync uses a CUDA event recorded after the H2D and
//! waited on by the compute stream just before the layer's first use.
//!
//! Public API surface mirrors the existing `VarBuilder` consumption
//! pattern in `streaming.rs::forward`, so the integration is a small
//! diff: `prefetch(i+1)` at the top of the loop, `take_var_builder(i)`
//! to consume the current layer.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::cuda_backend::{cudarc, CudaStorage};
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, Storage, Tensor};
use candle_nn::VarBuilder;
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::error::{Error, Result};

/// Per-tensor metadata extracted once from a sample layer header. All
/// layer shards in a Llama checkpoint have the same schema (same
/// tensor names + shapes + dtypes + relative offsets), so a single
/// schema parse describes every layer.
#[derive(Debug, Clone)]
struct TensorSchema {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    byte_offset: usize, // within the data section
    byte_len: usize,
}

/// Double-buffered async weight uploader.
pub struct AsyncWeightLoader {
    cuda_device: candle_core::cuda_backend::CudaDevice,
    ctx: Arc<CudaContext>,
    upload_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,

    /// `layer_data_bytes` is the size of the data section of one
    /// layer's safetensors file (header excluded). All layer files in
    /// a single model share the same size.
    layer_data_bytes: usize,
    /// Schema tensor names are the *sample layer's* full names
    /// (e.g. `model.layers.0.self_attn.q_proj.weight`). At
    /// `take_var_builder` time we rewrite the prefix to point at the
    /// caller's current layer index.
    sample_layer_idx: usize,
    schema: Vec<TensorSchema>,

    /// Mmap regions kept alive while their H2D is in flight. Indexed
    /// by slot — overwritten on next prefetch into that slot.
    inflight_mmap: [Option<Mmap>; 2],
    /// Device blobs — ping-pong.
    device: [CudaSlice<u8>; 2],
    /// Event recorded after each slot's H2D completes; compute stream
    /// waits on this before reading the slot.
    upload_events: [Option<CudaEvent>; 2],

    /// Which slot most-recently-finished is the "ready" one for `take`.
    ready_slot: usize,
}

impl AsyncWeightLoader {
    /// Construct from a sample layer file. The header of that file is
    /// parsed to extract the (name, dtype, shape, offset, length)
    /// schema applied to every layer.
    pub fn new(
        device: &Device,
        sample_layer_path: &Path,
        target_dtype: DType,
    ) -> Result<Self> {
        let cuda_device = match device {
            Device::Cuda(cd) => cd.clone(),
            _ => {
                return Err(Error::Msg(
                    "AsyncWeightLoader requires a CUDA device".into(),
                ))
            }
        };
        let compute_stream = cuda_device.cuda_stream();
        let ctx = compute_stream.context().clone();
        let upload_stream = ctx.new_stream().map_err(|e| Error::Msg(e.to_string()))?;

        // Parse the sample layer's header to derive the schema.
        let file = File::open(sample_layer_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(Error::Msg("safetensors file too small".into()));
        }
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_size;
        if mmap.len() < data_start {
            return Err(Error::Msg("safetensors header overflow".into()));
        }
        let data_section_len = mmap.len() - data_start;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| Error::Msg(format!("safetensors parse: {e}")))?;
        let mut schema = Vec::with_capacity(st.names().len());
        for name in st.names() {
            let view = st.tensor(name).map_err(|e| Error::Msg(format!("tensor {name}: {e}")))?;
            let shape: Vec<usize> = view.shape().to_vec();
            let byte_offset =
                view.data().as_ptr() as usize - (mmap.as_ptr() as usize + data_start);
            let byte_len = view.data().len();
            let dtype = match (view.dtype(), target_dtype) {
                (safetensors::Dtype::F16, _) => DType::F16,
                (safetensors::Dtype::BF16, _) => DType::BF16,
                (safetensors::Dtype::F32, _) => DType::F32,
                (other, _) => {
                    return Err(Error::Msg(format!(
                        "unsupported safetensors dtype {:?}",
                        other
                    )))
                }
            };
            schema.push(TensorSchema {
                name: name.to_string(),
                dtype,
                shape,
                byte_offset,
                byte_len,
            });
        }

        // Device ping-pong only. We no longer pre-allocate pinned host
        // buffers — the H2D reads directly from the mmap'd file region
        // (pageable host memory). The CUDA driver pays a small extra
        // bandwidth cost vs. truly pinned memory, but in exchange we
        // avoid a per-layer ~400 ms user-space memcpy that previously
        // serialised the prefetch path.
        let device_0 = unsafe { upload_stream.alloc::<u8>(data_section_len) }
            .map_err(|e| Error::Msg(e.to_string()))?;
        let device_1 = unsafe { upload_stream.alloc::<u8>(data_section_len) }
            .map_err(|e| Error::Msg(e.to_string()))?;

        // Derive the sample's layer index from the filename so we can
        // rewrite the per-tensor prefix to match the caller's layer at
        // `take_var_builder` time. Falls back to 0 if the filename
        // doesn't follow the `model.layers.N.safetensors` pattern.
        let sample_layer_idx = sample_layer_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("model.layers."))
            .and_then(|n| n.strip_suffix(".safetensors"))
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(0);

        Ok(Self {
            cuda_device,
            ctx,
            upload_stream,
            compute_stream,
            layer_data_bytes: data_section_len,
            sample_layer_idx,
            schema,
            inflight_mmap: [None, None],
            device: [device_0, device_1],
            upload_events: [None, None],
            ready_slot: usize::MAX, // nothing ready yet
        })
    }

    /// Schedule an H2D for `layer_path` into the inactive slot. The
    /// mmap is kept alive until the *next* `prefetch` (so the H2D has
    /// a stable host pointer to copy from). The H2D itself is async
    /// on the upload stream — it returns immediately even if the DMA
    /// hasn't started; an event is recorded so the compute stream can
    /// wait on it later.
    pub fn prefetch(&mut self, layer_path: &Path) -> Result<usize> {
        let target = if self.ready_slot == usize::MAX {
            0
        } else {
            1 - self.ready_slot
        };

        // mmap source file. The mmap region lives in the page cache;
        // the kernel may DMA from it directly (pageable host memory),
        // avoiding the 400 ms-per-layer pinned-staging memcpy that was
        // serialising the prefetch path.
        let file = File::open(layer_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_size;
        let data_section = &mmap[data_start..];
        if data_section.len() != self.layer_data_bytes {
            return Err(Error::Msg(format!(
                "layer size mismatch: schema expects {} bytes, {} has {}",
                self.layer_data_bytes,
                layer_path.display(),
                data_section.len()
            )));
        }

        // Async H2D directly from the (pageable) mmap region. The CUDA
        // driver internally pins / stages as needed but without our
        // extra user-space memcpy. Keep the mmap alive in `inflight`
        // until the next prefetch overwrites it — the H2D may still
        // be running when this function returns.
        self.upload_stream
            .memcpy_htod(data_section, &mut self.device[target])
            .map_err(|e| Error::Msg(e.to_string()))?;
        self.inflight_mmap[target] = Some(mmap);

        // Record an event so the compute stream can wait on it later.
        let event = self
            .upload_stream
            .record_event(None)
            .map_err(|e| Error::Msg(e.to_string()))?;
        self.upload_events[target] = Some(event);
        self.ready_slot = target;
        Ok(target)
    }

    /// Sync on the most recent prefetch and return a `VarBuilder`
    /// pointing at the layer's tensors, with names rewritten so that
    /// the per-layer prefix matches `layer_idx`. Callers then consume
    /// via `vb.pp(format!("model.layers.{layer_idx}"))` exactly as in
    /// the synchronous mmap path.
    ///
    /// Internally:
    ///   1. compute_stream.wait_event(upload_event)  — cross-stream sync
    ///   2. for each tensor in schema: D2D copy bytes into a typed
    ///      CudaSlice of the right dtype, wrap as candle Tensor
    ///   3. assemble HashMap<rewritten_name, Tensor>, return VarBuilder
    pub fn take_var_builder<'a>(
        &mut self,
        target_dtype: DType,
        layer_idx: usize,
    ) -> Result<VarBuilder<'a>> {
        if self.ready_slot == usize::MAX {
            return Err(Error::Msg("no prefetch issued before take_var_builder".into()));
        }
        let slot = self.ready_slot;
        let event = self.upload_events[slot]
            .as_ref()
            .ok_or_else(|| Error::Msg("ready_slot has no event".into()))?;
        // Compute stream waits on the upload's event — this is the
        // cross-stream synchronisation that lets us safely consume
        // device[slot] from compute kernels.
        self.compute_stream
            .wait(event)
            .map_err(|e| Error::Msg(e.to_string()))?;

        // Reconstruct each tensor as a typed CudaSlice + candle Tensor.
        let device_handle = Device::Cuda(self.cuda_device.clone());
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(self.schema.len());
        for s in &self.schema {
            let byte_view = self.device[slot].slice(s.byte_offset..s.byte_offset + s.byte_len);
            let tensor = match s.dtype {
                DType::F16 => {
                    let n = s.byte_len / 2;
                    let mut dst = unsafe { self.upload_stream.alloc::<half::f16>(n) }
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    // SAFETY: f16 has no padding bytes; reinterpreting
                    // n*2 bytes as n f16 elements is layout-equivalent.
                    let mut dst_u8 = unsafe { dst.transmute_mut::<u8>(s.byte_len) }
                        .ok_or_else(|| Error::Msg("transmute_mut size mismatch".into()))?;
                    self.upload_stream
                        .memcpy_dtod(&byte_view, &mut dst_u8)
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    drop(dst_u8);
                    let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(
                        dst,
                        self.cuda_device.clone(),
                    ));
                    Tensor::from_storage(storage, s.shape.clone(), BackpropOp::none(), false)
                }
                DType::F32 => {
                    let n = s.byte_len / 4;
                    let mut dst = unsafe { self.upload_stream.alloc::<f32>(n) }
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    let mut dst_u8 = unsafe { dst.transmute_mut::<u8>(s.byte_len) }
                        .ok_or_else(|| Error::Msg("transmute_mut size mismatch".into()))?;
                    self.upload_stream
                        .memcpy_dtod(&byte_view, &mut dst_u8)
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    drop(dst_u8);
                    let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(
                        dst,
                        self.cuda_device.clone(),
                    ));
                    Tensor::from_storage(storage, s.shape.clone(), BackpropOp::none(), false)
                }
                other => {
                    return Err(Error::Msg(format!("dtype not handled: {:?}", other)))
                }
            };
            // Coerce to the target dtype if it differs (e.g., F32 inv_freq
            // in an F16 model). candle Linear/RmsNorm load paths cast
            // anyway, but doing it here keeps the VarBuilder consistent.
            let tensor = if tensor.dtype() == target_dtype {
                tensor
            } else {
                tensor.to_dtype(target_dtype)?
            };
            // Rewrite the per-tensor name so the cached layer-N schema
            // matches the caller's current layer index. We only swap
            // the literal `model.layers.{sample_idx}.` prefix; tensors
            // without that prefix (none in current llama files but
            // future-proof) pass through unchanged.
            let from = format!("model.layers.{}.", self.sample_layer_idx);
            let to = format!("model.layers.{layer_idx}.");
            let rewritten = if let Some(rest) = s.name.strip_prefix(&from) {
                format!("{to}{rest}")
            } else {
                s.name.clone()
            };
            tensors.insert(rewritten, tensor);
        }

        // VarBuilder::from_tensors wraps the HashMap so Block::load can
        // use the same code path as the synchronous mmap-backed VB.
        Ok(VarBuilder::from_tensors(tensors, target_dtype, &device_handle))
    }
}

/// Convenience constructor that picks the first layer file from a
/// directory as the schema sample.
pub fn from_layer_dir(
    device: &Device,
    layer_dir: &Path,
    target_dtype: DType,
) -> Result<(AsyncWeightLoader, Vec<PathBuf>)> {
    let mut layer_files: Vec<(usize, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(layer_dir)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            Error::Msg("non-utf8 filename in layer dir".into())
        })?;
        if let Some(idx_str) = name
            .strip_prefix("model.layers.")
            .and_then(|s| s.strip_suffix(".safetensors"))
        {
            if let Ok(idx) = idx_str.parse::<usize>() {
                layer_files.push((idx, entry.path()));
            }
        }
    }
    if layer_files.is_empty() {
        return Err(Error::Msg(format!(
            "no model.layers.*.safetensors in {}",
            layer_dir.display()
        )));
    }
    layer_files.sort_by_key(|(i, _)| *i);
    let sample = layer_files[0].1.clone();
    let loader = AsyncWeightLoader::new(device, &sample, target_dtype)?;
    let paths: Vec<PathBuf> = layer_files.into_iter().map(|(_, p)| p).collect();
    Ok((loader, paths))
}
