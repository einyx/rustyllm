//! Phase 2 spike — proves we can hand a manually-uploaded `CudaSlice`
//! to candle as a `Tensor` and use it in a normal matmul. This is the
//! critical bridge for async H2D: if it works, we can do the H2D
//! ourselves on a separate stream and still feed the result into
//! candle's standard `Linear::forward` path.
//!
//! Path under test:
//!   user-controlled cudarc alloc → memcpy_htod → CudaSlice
//!     → CudaStorage::wrap_cuda_slice
//!     → Tensor::from_storage
//!     → tensor.matmul(...)
//!
//! Reference path (oracle):
//!   Tensor::new(host_vec, device) → tensor.matmul(...)
//!
//! Both should produce numerically identical results.

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

    // 2x3 weight matrix, known values so we can spot bugs in dequant/layout.
    const ROWS: usize = 2;
    const COLS: usize = 3;
    let host_data: Vec<f16> = (0..ROWS * COLS)
        .map(|i| f16::from_f32(i as f32 + 1.0))
        .collect();
    // Visualised:
    //   [[1, 2, 3],
    //    [4, 5, 6]]

    let device = Device::new_cuda(0)?;
    println!("device: {:?}", device);

    // === Reference: build the weight tensor via candle's normal API ===
    let ref_tensor = Tensor::from_vec(host_data.clone(), (ROWS, COLS), &device)?;
    println!("reference tensor built via Tensor::from_vec: {:?}", ref_tensor.shape());

    // === Under test: build it through manual cudarc alloc + wrap ===
    let cuda_device = match &device {
        Device::Cuda(cd) => cd.clone(),
        _ => unreachable!("requested CUDA above"),
    };

    // Get the candle device's underlying cudarc context. Allocating on
    // *this* context is essential — pointers from a different context
    // would dangle when handed to candle's kernels.
    let stream = cuda_device.cuda_stream();
    let ctx = stream.context().clone();
    println!("got cudarc context from candle (Arc<CudaContext> via CudaStream::context)");

    // Allocate device buffer using the candle-owned context's stream,
    // then sync-memcpy the host data in. In the real implementation this
    // would be an async memcpy on a SEPARATE stream that was created
    // via ctx.new_stream(), with proper event sync before consumption.
    let mut dev_slice: cudarc::driver::CudaSlice<f16> =
        unsafe { stream.alloc::<f16>(host_data.len()) }?;
    stream.memcpy_htod(&host_data, &mut dev_slice)?;
    println!("manually uploaded {} f16 elements to device", host_data.len());

    // The bridge: wrap CudaSlice → CudaStorage → Storage → Tensor.
    let cuda_storage = CudaStorage::wrap_cuda_slice(dev_slice, cuda_device.clone());
    let storage = Storage::Cuda(cuda_storage);
    let our_tensor = Tensor::from_storage(
        storage,
        (ROWS, COLS),
        BackpropOp::none(),
        /* is_variable */ false,
    );
    println!("constructed Tensor::from_storage with shape {:?}, dtype {:?}",
             our_tensor.shape(), our_tensor.dtype());

    // === Validate: read back contents and compare ===
    let ref_read: Vec<f16> = ref_tensor.flatten_all()?.to_vec1::<f16>()?;
    let our_read: Vec<f16> = our_tensor.flatten_all()?.to_vec1::<f16>()?;
    println!("\nreference contents: {:?}", ref_read);
    println!("our contents:       {:?}", our_read);
    assert_eq!(ref_read, our_read, "Tensor contents differ between paths");

    // === Validate: run an actual op (matmul) through both ===
    // Build a 3x4 RHS so the matmul output is 2x4.
    let rhs_host: Vec<f16> = (0..COLS * 4).map(|i| f16::from_f32((i % 3) as f32 - 1.0)).collect();
    let rhs = Tensor::from_vec(rhs_host, (COLS, 4), &device)?;

    let ref_out = ref_tensor.matmul(&rhs)?;
    let our_out = our_tensor.matmul(&rhs)?;
    let ref_vec: Vec<Vec<f16>> = ref_out.to_vec2::<f16>()?;
    let our_vec: Vec<Vec<f16>> = our_out.to_vec2::<f16>()?;
    println!("\nreference matmul result: {:?}", ref_vec);
    println!("our matmul result:       {:?}", our_vec);
    assert_eq!(ref_vec, our_vec, "matmul outputs differ");

    println!("\n✓ Phase 2 confirmed: manually-uploaded CudaSlice round-trips through");
    println!("  candle's Tensor::from_storage and produces matmul-identical results.");
    println!("  → ready to wire async H2D into the streaming Block path.");
    Ok(())
}
