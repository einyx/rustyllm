//! Async H2D spike — validates the cudarc multi-stream + pinned-host
//! pattern that the streaming Block needs to overlap PCIe transfer of
//! layer N+1 with GPU compute on layer N.
//!
//! See `design_async_h2d.md` for the full design. This binary just
//! answers: can we, in cudarc 0.17 + candle 0.10, allocate pinned host
//! memory, create a non-default CUDA stream, issue an async memcpy on
//! it, and measure overlap with work on a different stream?
//!
//! Run: `cargo run --release --features cuda --example async_h2d_spike`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This spike requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::cuda_backend::cudarc;
    use cudarc::driver::CudaContext;
    use std::time::Instant;

    // Layer-size H2D buffer. Real Llama-70B layers are ~1.6 GB; we
    // use 256 MiB here so the spike fits alongside other GPU users on
    // the same card. Bandwidth measurement is still meaningful — the
    // PCIe pipeline saturates at >64 MiB.
    const LAYER_BYTES: usize = 256 << 20; // 256 MiB

    let ctx = CudaContext::new(0)?;
    let compute = ctx.default_stream();
    let upload = ctx.new_stream()?;
    println!("ctx: ordinal=0, two streams established (compute = default, upload = new)");

    // Pinned host buffers (cuMemAllocHost). Required for true async
    // DMA — pageable memory forces a staging copy in the CUDA driver.
    let mut host_pinned = unsafe { ctx.alloc_pinned::<u8>(LAYER_BYTES) }?;
    // Fill with a recognisable pattern so we can verify the round-trip.
    {
        let slice = host_pinned.as_mut_slice()?;
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
    }
    println!("host:  pinned buffer of {} MiB allocated + populated", LAYER_BYTES >> 20);

    // Two device buffers — the ping-pong target.
    let mut dev_a: cudarc::driver::CudaSlice<u8> = upload.alloc_zeros::<u8>(LAYER_BYTES)?;
    let mut dev_b: cudarc::driver::CudaSlice<u8> = upload.alloc_zeros::<u8>(LAYER_BYTES)?;
    println!("dev:   two device buffers of {} MiB each", LAYER_BYTES >> 20);

    // Baseline 1: synchronous H2D on the compute stream (mirrors what
    // candle does today inside `Block::load`).
    ctx.synchronize()?;
    let t = Instant::now();
    compute.memcpy_htod(&host_pinned, &mut dev_a)?;
    compute.synchronize()?;
    let sync_ms = t.elapsed().as_secs_f64() * 1e3;
    let bw_gibs = (LAYER_BYTES as f64 / (1 << 30) as f64) / (sync_ms / 1e3);
    println!(
        "\nsync H2D on default stream: {:6.1} ms   ({:.2} GiB/s)",
        sync_ms, bw_gibs
    );

    // Baseline 2: same memcpy but issued on the upload stream, then
    // immediately wait. Should be the same wall time — confirms no
    // hidden cost from using a non-default stream.
    ctx.synchronize()?;
    let t = Instant::now();
    upload.memcpy_htod(&host_pinned, &mut dev_b)?;
    upload.synchronize()?;
    let upload_only_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "sync H2D on upload stream:  {:6.1} ms   ({:.2} GiB/s)",
        upload_only_ms,
        (LAYER_BYTES as f64 / (1 << 30) as f64) / (upload_only_ms / 1e3)
    );

    // Overlap test: kick off H2D on upload stream, simulate compute
    // by doing a long device-to-device copy on the compute stream, and
    // measure total wall time. If overlap works, total < sync_ms +
    // compute_only_ms.
    //
    // The "compute" here is a 512 MiB D2D copy — keeps the compute
    // stream busy long enough to overlap with the LAYER_BYTES H2D
    // (which is ~10 ms on Gen4 PCIe at 25 GiB/s).
    const COMPUTE_BYTES: usize = 512 << 20;
    let mut compute_src: cudarc::driver::CudaSlice<u8> =
        compute.alloc_zeros::<u8>(COMPUTE_BYTES)?;
    let mut compute_dst: cudarc::driver::CudaSlice<u8> =
        compute.alloc_zeros::<u8>(COMPUTE_BYTES)?;

    // Loop D2D enough times to land near the H2D wall (~15 ms). A
    // single 512 MiB D2D on GDDR6 internal bandwidth is ~2 ms, so 10
    // iters ≈ 20 ms — a realistic stand-in for a transformer block's
    // compute window.
    const COMPUTE_ITERS: usize = 10;
    let do_compute = |stream: &std::sync::Arc<cudarc::driver::CudaStream>,
                      src: &cudarc::driver::CudaSlice<u8>,
                      dst: &mut cudarc::driver::CudaSlice<u8>|
     -> Result<(), cudarc::driver::DriverError> {
        for _ in 0..COMPUTE_ITERS {
            stream.memcpy_dtod(src, dst)?;
        }
        Ok(())
    };

    // Measure compute alone first.
    ctx.synchronize()?;
    let t = Instant::now();
    do_compute(&compute, &compute_src, &mut compute_dst)?;
    compute.synchronize()?;
    let compute_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "compute alone (D2D × {} × {} MiB):  {:6.1} ms",
        COMPUTE_ITERS,
        COMPUTE_BYTES >> 20,
        compute_ms
    );

    // Now run them in parallel: H2D on upload, D2D loop on compute.
    ctx.synchronize()?;
    let t = Instant::now();
    upload.memcpy_htod(&host_pinned, &mut dev_a)?;
    do_compute(&compute, &compute_src, &mut compute_dst)?;
    // Wait for both.
    upload.synchronize()?;
    compute.synchronize()?;
    let overlap_ms = t.elapsed().as_secs_f64() * 1e3;

    let serial_estimate = sync_ms + compute_ms;
    let savings = serial_estimate - overlap_ms;
    let pct = 100.0 * savings / serial_estimate;
    println!(
        "\nparallel H2D + compute:      {:6.1} ms",
        overlap_ms,
    );
    println!("if fully serial we'd expect: {:6.1} ms", serial_estimate);
    println!("savings:                    {:6.1} ms ({:.1}% reduction)", savings, pct);

    if savings > 5.0 {
        println!("\n✓ overlap confirmed — async H2D pattern works in cudarc 0.17.");
        println!("  next: wrap CudaSlice in candle CudaStorage via wrap_cuda_slice,");
        println!("  then feed it into Block::load instead of mmap'd safetensors.");
    } else {
        println!("\n⚠ no measurable overlap — investigate stream/event semantics.");
    }

    // Suppress "unused" warnings on dev_a/dev_b: keep them in scope until
    // the end so the OS doesn't reclaim the device pages during timing.
    drop((dev_a, dev_b, compute_src, compute_dst, host_pinned));
    Ok(())
}
