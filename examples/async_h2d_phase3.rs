//! Phase 3a spike — async-upload a real layer `.safetensors` blob to
//! the GPU on a separate stream, parse the safetensors header, and
//! verify per-tensor offset math by spot-checking one named tensor
//! against the reference (`VarBuilder::from_mmaped_safetensors`) path.
//!
//! Establishes the contract for Phase 3b (`AsyncWeightLoader` wired
//! into `streaming.rs::forward`):
//!   - One device alloc per layer (sized to the tensor-data section)
//!   - Async H2D issued on a stream distinct from candle's compute
//!     stream — overlap with current-layer compute
//!   - Header parse + per-tensor D2D copy into typed CudaSlices
//!   - Each typed slice wraps to a candle Tensor with the right dtype
//!     and shape

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This spike requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::cuda_backend::{cudarc, CudaStorage};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Device, Storage, Tensor};
    use half::f16;
    use safetensors::SafeTensors;
    use std::time::Instant;

    const LAYER_PATH: &str =
        "/mnt/4t/cache/airllm_splits/splitted_model/model.layers.0.safetensors";
    const SPOT_CHECK: &str = "model.layers.0.input_layernorm.weight"; // 8192 F16

    // ─── 1. read the layer file (page cache likely warm) ──────────────
    let bytes = std::fs::read(LAYER_PATH)?;
    println!("layer file: {} ({:.2} GiB)", LAYER_PATH, bytes.len() as f64 / (1u64 << 30) as f64);

    let st = SafeTensors::deserialize(&bytes)?;
    println!("tensors: {} named entries", st.names().len());

    let header_size = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let data_section = &bytes[8 + header_size..];

    // ─── 2. candle device + new upload stream from same context ───────
    let device = Device::new_cuda(0)?;
    let cuda_device = match &device {
        Device::Cuda(cd) => cd.clone(),
        _ => unreachable!(),
    };
    let ctx = cuda_device.cuda_stream().context().clone();
    let upload_stream = ctx.new_stream()?;

    // ─── 3. pinned staging + async H2D of the whole data section ──────
    let t = Instant::now();
    let mut pinned = unsafe { ctx.alloc_pinned::<u8>(data_section.len()) }?;
    pinned.as_mut_slice()?.copy_from_slice(data_section);
    let stage_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut dev_blob: cudarc::driver::CudaSlice<u8> =
        unsafe { upload_stream.alloc::<u8>(data_section.len()) }?;
    upload_stream.memcpy_htod(&pinned, &mut dev_blob)?;
    upload_stream.synchronize()?;
    let h2d_ms = t.elapsed().as_secs_f64() * 1e3;
    let bw = (data_section.len() as f64 / (1u64 << 30) as f64) / (h2d_ms / 1e3);
    println!(
        "host → pinned: {:.1} ms   pinned → device: {:.1} ms ({:.2} GiB/s)",
        stage_ms, h2d_ms, bw
    );

    // ─── 4. parse header + slice out the spot-check tensor ────────────
    let view = st.tensor(SPOT_CHECK)?;
    let shape: Vec<usize> = view.shape().to_vec();
    let n_elems: usize = shape.iter().product();
    // safetensors 0.4 doesn't expose data_offsets directly; compute the
    // offset by pointer arithmetic within the source byte slice. Both
    // `view.data()` and `data_section` are views into the same `bytes`
    // Vec, so this is well-defined.
    let start =
        view.data().as_ptr() as usize - data_section.as_ptr() as usize;
    let end = start + view.data().len();
    assert_eq!(end - start, n_elems * 2, "F16 byte length");
    println!(
        "\nspot-check tensor: {SPOT_CHECK}\n  shape={:?}, F16, bytes {}..{}",
        shape, start, end
    );

    // D2D copy of just this tensor's bytes from the big blob into a
    // typed CudaSlice<f16>. In production we'd avoid this copy by
    // constructing a typed view directly, but the cudarc safe API
    // doesn't expose typed reinterpretation of an owned u8 slice
    // without a copy — the D2D fix-up is small (~16 KB here) and
    // pipelined with the H2D upload on the same upload stream.
    let byte_view = dev_blob.slice(start..end);
    let dev_f16 = {
        let mut dst = unsafe { upload_stream.alloc::<f16>(n_elems) }?;
        // f16 is 2 bytes; the bytes in `byte_view` are little-endian.
        // We get a CudaSlice<f16> by copying the bytes into a fresh
        // typed alloc via dtod on the upload stream.
        // SAFETY: f16 is 2 bytes with no internal padding; reinterpreting
        // n_elems f16s as 2*n_elems u8s is valid round-trip-equivalent
        // memory access, and the slice never escapes this scope.
        let mut dst_u8 = unsafe { dst.transmute_mut::<u8>(n_elems * 2) }.unwrap();
        upload_stream.memcpy_dtod(&byte_view, &mut dst_u8)?;
        drop(dst_u8);
        upload_stream.synchronize()?;
        dst
    };

    // ─── 5. wrap as candle Tensor ─────────────────────────────────────
    let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(dev_f16, cuda_device.clone()));
    let our_tensor = Tensor::from_storage(storage, shape.clone(), BackpropOp::none(), false);

    // ─── 6. reference path via VarBuilder ─────────────────────────────
    use candle_nn::VarBuilder;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[std::path::PathBuf::from(LAYER_PATH)],
            DType::F16,
            &device,
        )?
    };
    let ref_tensor = vb.get_with_hints(&shape[..], SPOT_CHECK, candle_nn::Init::Const(0.0))?;

    // ─── 7. compare ───────────────────────────────────────────────────
    let ours: Vec<f16> = our_tensor.flatten_all()?.to_vec1::<f16>()?;
    let theirs: Vec<f16> = ref_tensor.flatten_all()?.to_vec1::<f16>()?;
    assert_eq!(ours.len(), theirs.len());
    let mismatches = ours
        .iter()
        .zip(theirs.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "comparison: {} elements, {} mismatches",
        ours.len(),
        mismatches
    );
    assert_eq!(mismatches, 0, "byte mismatch — offset math is wrong");
    println!(
        "  sample (first 6): ours={:?}\n                    theirs={:?}",
        &ours[..6.min(ours.len())],
        &theirs[..6.min(theirs.len())],
    );

    println!("\n✓ Phase 3a confirmed: async-uploaded 1.6 GiB layer blob, header parse + offset");
    println!("  slicing produces byte-identical F16 tensor via the candle Tensor bridge.");
    println!("  H2D bandwidth: {:.1} GiB/s — leaves headroom on Gen4 x16 (31.5 GiB/s peak).", bw);
    Ok(())
}
