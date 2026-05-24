# RustyLLM

Run 70B LLMs on a single small GPU. Pure Rust, no Python, no BLAS, no sudo.

The idea: don't load the whole model. Split each transformer block across GPU + host RAM + on-demand disk streaming, and keep only what fits on the device resident.

```
GPU (12 GB)    │ embed + lm_head + 20 hot layers (Q4)
Host RAM       │ 60 warm layers (Q4)
Disk shards    │ rest, streamed per token (when needed)
```

On a single RTX 4070 (12 GB) + 125 GB RAM, **70B inference runs at 0.75 tok/s — 1.5× faster than Ollama on the same hardware** while using less VRAM.

## Quick start

```bash
# Build (pick one backend)
cargo build --release                          # CPU only
cargo build --release --features cuda          # NVIDIA
cargo build --release --features metal         # Apple Silicon
cargo build --release --features cuda,parallel-cpu  # NVIDIA + 32-core CPU

# Run 70B inference (downloads weights on first use, ~140 GB for F16)
cargo run --release --features cuda --example bench_streaming_70b_q4
```

That's it. RustyLLM auto-shards the HF model on first load into per-layer files (one-time cost) and caches under `~/.cache/huggingface/`.

## Library

Quantized hybrid (recommended for speed on small GPUs):

```rust
use rustyllm::{StreamingLlamaQuantized, inference::LoadOptions};
use std::path::Path;

let mut model = StreamingLlamaQuantized::from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    Path::new("/path/to/q4k_shards"),
    LoadOptions::default(),
)?;
model.pin_resident_layers(20)?;       // keep 20 layers on GPU
model.pin_cpu_resident_layers(60)?;   // keep 60 in host RAM
let text = model.generate("What is the capital of France?", 32, 0.0, None)?;
```

F16 (full precision, slower):

```rust
use rustyllm::{StreamingLlama, inference::LoadOptions};

let model = StreamingLlama::from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    LoadOptions::default(),
)?;
let text = model.generate("...", 32, 0.0, None)?;
```

Family auto-detect (Llama / Mistral / Qwen2 / InternLM2):

```rust
use rustyllm::{StreamingFamily, inference::LoadOptions};
let model = StreamingFamily::from_pretrained("model-id", LoadOptions::default())?;
```

## Supported models

- **Llama**, **Mistral**, **Qwen2**, **InternLM2** — auto-dispatched via `StreamingFamily`
- **Mixtral 8x7B** — sparse MoE, `StreamingMixtral` (manual dispatch)
- **ChatGLM** — 2D RoPE + SwiGLU, `StreamingChatGlm`
- **Baichuan-13B** — ALiBi attention, `StreamingBaichuan`

Verified end-to-end on **Platypus2-70B-instruct** with output byte-identical to Python AirLLM.

## Benchmarks

Platypus2-70B-instruct, 16 tokens greedy, prompt `"What is the capital of the United States, and what river runs through it?"`. Hardware: RTX 4070 (12 GB) + i9-13900K (32 threads) + 125 GB RAM.

| Configuration | Quant | Wall | tok/s | Peak GPU | Peak Host |
|---|---|---:|---:|---:|---:|
### 70B — Q4 (default speed/quality tradeoff, ~38 GB on disk)

| Configuration                                                | Quant  | Wall    | tok/s | Peak GPU | Peak Host |
|---|---|---:|---:|---:|---:|
| **rustyllm Q4 spec decoding (7B draft, K=2, adaptive)**      | Q4_K   | **14.0 s** | **1.21** | **10.7 GB** | **32 GB** |
| rustyllm Q4 hybrid (20 GPU + 60 CPU resident)                | Q4_K   | 21.4 s  | 0.75  | 11.4 GB  | 28 GB |
| rustyllm Q4 hybrid (16 GPU + 64 CPU resident)                | Q4_K   | 23.1 s  | 0.69  | 9.5 GB   | 28 GB |
| Ollama llama3.3:70b                                          | Q4_K_M | 34 s    | 0.47  | 10.9 GB  | — |
| rustyllm Q4 hybrid (20 GPU + 60 CPU, Q4_0)                   | Q4_0   | 42.1 s  | 0.38  | 11.4 GB  | 28 GB |
| rustyllm Q4 + 16 GPU resident, 64 streamed                   | Q4_K   | 208 s   | 0.077 | 10.0 GB  | 5 GB |
| rustyllm Q4 pure streaming                                   | Q4_K   | 427 s   | 0.037 | 2.8 GB   | 5 GB |

### 70B — Q3 (lower-memory alternative, ~29 GB on disk)

| Configuration                                                | Quant  | Wall    | tok/s | Peak GPU | Peak Host |
|---|---|---:|---:|---:|---:|
| rustyllm Q3 hybrid (26 GPU + 54 CPU resident)                | Q3_K   | 21.6 s  | 0.74  | 11.7 GB  | **19 GB** |

### 70B — F16 (full precision, ~140 GB on disk)

| Configuration                                                | Quant  | Wall     | tok/s  | Peak GPU | Peak Host |
|---|---|---:|---:|---:|---:|
| **rustyllm F16/Q4 mixed (F16 resident, 16 streamed Q4)**     | F16/Q4 | **178.9 s** | **0.089** | **7.1 GB** | **98 GB** |
| rustyllm F16 hybrid (4 GPU + 60 CPU + parallel-cpu)          | F16    | 320.8 s  | 0.050  | 7.1 GB   | 99 GB |
| rustyllm F16 hybrid (4 GPU + 60 CPU)                         | F16    | 405 s    | 0.040  | 7.1 GB   | 99 GB |
| rustyllm F16 streaming + async H2D                           | F16    | 723 s    | 0.022  | 6.1 GB   | 5 GB |
| rustyllm F16 streaming (sync)                                | F16    | 1033 s   | 0.015  | 2.9 GB   | 5 GB |
| Python AirLLM (CUDA)                                         | F16    | 1524 s   | 0.011  | 1.9 GB   | — |
| rustyllm F16 streaming (pure CPU)                            | F16    | 1798 s   | 0.009  | —        | 5 GB |

### 7B — Q4 (used as the speculative-decoding draft)

| Configuration                                                | Quant  | Per-token | Notes |
|---|---|---:|---|
| 32 layers all GPU resident                                   | Q4_K   | ~50 ms   | measured as draft during spec K=2 runs (~20 tok/s standalone) |
| 32 layers all CPU resident (no GPU)                          | Q4_K   | ~200 ms  | extrapolated from CPU residency cost on 70B (~5 tok/s) |

7B F16 (~14 GB) exceeds 12 GB VRAM, so all-GPU residency isn't possible on this card; would need ≥16 GB device.

### 70B on Apple Silicon (M3 Pro, 36 GB unified)

Tighter budget than the Linux box: 36 GB has to hold model + activations + draft + OS. No discrete VRAM ceiling (Metal allocations come out of the same pool as host RAM), but the total is the constraint. Same Platypus2-70B-instruct, same prompt, same 7B draft.

| Configuration                                                  | Quant   | tok/s | Notes |
|---|---|---:|---|
| **Spec K=12, Q2K target + 40 layers pinned**                   | Q2_K    | **0.240** | best speed; "Anacostia" answer (also valid) |
| Spec K=12, Q3K-MLP/Q4K-attn target + 40 pin                    | mixed   | 0.187 | "Potomac" — better answer, ~22 % slower |
| Spec K=12, Q4K target streamed                                 | Q4_K    | 0.115 | baseline-quality, no pinning needed |
| Q4K streamed (no spec)                                         | Q4_K    | 0.033 | baseline |
| 7B Q4K fully resident (reference for "fast Metal")             | Q4_K    | 18.65 | proves Metal kernels are fast when weights fit |

**The Apple-Silicon-specific finding:** spec decoding with an aggressively-quantized target works even when the target wouldn't free-generate well alone. Q2K-70B alone produces garbage ("Theʎʎʎʎʎʎ…"), but as the *verifier* it just argmax-checks the 7B draft's proposals — and Q2K's argmax mostly agrees with Q4K's on common-vocabulary completions. Quality varies by prompt type:

| Prompt type      | Q2K target tok/s | accept rate | quality vs Q4K target |
|---|---:|---:|---|
| Factual lookup    | 0.240            | 44 %        | equivalent (different but valid completions) |
| Reasoning         | 0.159            | 23 %        | worse — Q2K argmax wobble shortens answers |
| Code generation   | 0.124            | 19 %        | equivalent (both struggle; base model is mediocre at code) |

Use Q2K target when speed > nuance (factual chat, lookup tools). Use Q4K target when you need full quality (reasoning, long-form).

Recipe (assumes both shard sets prepared via `ensure_q4_shards`):

```bash
RUSTYLLM_TARGET_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-70B-instruct_q2k \
RUSTYLLM_DRAFT_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-7B_q4k \
RUSTYLLM_TARGET_GPU=40 RUSTYLLM_SPEC_K=12 RUSTYLLM_MAX_TOKENS=64 \
  cargo run --release --features metal --example spec_decode_70b
```

To produce a custom mixed-quant target (attn Q4K + MLP Q3K, ~30 GB) from existing Q4K shards without re-downloading 140 GB of F16:

```bash
RUSTYLLM_SRC_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-70B-instruct_q4k \
RUSTYLLM_DST_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-70B-instruct_attnq4_mlpq3 \
RUSTYLLM_ATTN_QUANT=q4k RUSTYLLM_MLP_QUANT=q3k \
  cargo run --release --example requantize_mixed
```

Output of Q2K-target and Q3K-MLP-target spec runs is *not* byte-identical to Q4K — the verifier itself is a lossier model — but the speculative-decoding contract still holds: every committed token is target's argmax at that position. So output equals what greedy Q2K (or Q3K-MLP) target would produce on its own.

All other rustyllm runs (Q4K and above) produce output byte-identical to the F16 reference continuation.

**Speculative decoding** (`examples/spec_decode_70b.rs`): a small Llama-2-7B "draft" model proposes K tokens, the full 70B "target" verifies them in a single batched forward. The verifier is always the target, so output is mathematically identical to greedy decoding. **Adaptive K** (default): start at K=2, drop K when accept rate falls below 40% (eventually to K=0 = pure baseline), grow back when it recovers. This way speculation never meaningfully regresses on unpredictable prompts:

| Prompt                                              | Baseline             | Adaptive spec        | Speedup | Notes |
|---|---:|---:|---:|---|
| Short factual ("What is the capital of the US…?")   | 22.7 s / 17 tok      | 14.3 s / 17 tok      | **1.60×** | stable K=2, 83 % accept |
| Math reasoning (step-by-step)                       | 67.8 s / 48 tok      | 50.0 s / 49 tok      | **1.39×** | adapts between K=1/2     |
| Code generation                                     | 76.4 s / 64 tok      | 65.3 s / 67 tok      | **1.22×** | mostly K=2, ~68 % accept |
| Translation                                         | 51.0 s / 48 tok      | 47.4 s / 51 tok      | **1.15×** | stable K=2, 73 % accept  |
| Free-form explanation                               | 48.4 s / 48 tok      | 52.1 s / 50 tok      | **0.97×** | drops to K=0, near-break-even |

Speculation wins biggest on templated / predictable content. On open-ended generation the adaptive controller correctly drops K to 0 within a few rounds, recovering near-baseline cost (vs 0.81× regression with fixed K=2).

Disable adaptation with `RUSTYLLM_ADAPTIVE_K=0`. Set max K with `RUSTYLLM_SPEC_K=N`. Use custom prompts via `RUSTYLLM_PROMPT="..." RUSTYLLM_MAX_TOKENS=N`.

**Three things to read off this table:**

1. **Q4 hybrid beats Ollama 1.5× on identical hardware** while using less VRAM (9.5 GB vs 10.9 GB). The combination of per-layer GGUF loading + GPU/CPU residency tiers is what unlocks it.
2. **Mixed-precision streaming is novel** — F16 weights for the resident tiers (per-token compute) plus Q4 for the streamed tier (PCIe-bound). Cuts F16 wall time 1.8× with byte-identical output, because the 20 % of weights that bottleneck on PCIe become 3× cheaper to transfer while the 80 % the model touches every token keep full precision.
3. **Each layer matters**: F16 → Q4 (2.5×), pure streaming → residency (8×), CUDA-only → CUDA+CPU hybrid (9×). The hybrid step is the big one: all 80 layers stay in RAM, none stream from disk per token.

## Tuning

The big knobs are environment variables — pick what fits your VRAM and host RAM:

```bash
# Q4 hybrid (recommended starting point)
RUSTYLLM_RESIDENT_LAYERS=20 RUSTYLLM_CPU_RESIDENT_LAYERS=60 \
  cargo run --release --features cuda --example bench_streaming_70b_q4

# F16 mixed-precision (full precision per-token, Q4 only where PCIe-bound)
RUSTYLLM_RESIDENT_LAYERS=4 RUSTYLLM_CPU_RESIDENT_LAYERS=60 \
RUSTYLLM_Q_STREAM_DIR=/path/to/q4_shards \
  cargo run --release --features cuda,parallel-cpu --example bench_streaming_70b

# Pure streaming (lowest VRAM, ~3 GB)
cargo run --release --features cuda --example bench_streaming_70b_q4
```

Sweet spots on a 12 GB / 125 GB host (RTX 4070 / 32-core CPU):

| | Q4 hybrid | F16 mixed | F16 hybrid | F16 streaming |
|---|---|---|---|---|
| GPU residency | **20** | **4** | **4** | 0 |
| CPU residency | **60** | **60** | **60** | 0 |
| Streamed | 0 | 16 (Q4) | 16 (F16) | all 80 (F16) |

If you have more VRAM (24 GB+): increase `RUSTYLLM_RESIDENT_LAYERS`. The Q4 ceiling is 20 layers per 12 GB; F16 is 4 layers per 12 GB (each F16 layer is ~1.75 GB, each Q4_K layer is ~0.6 GB).

## Tooling

```bash
# Split an HF model into per-layer F16 shards (one-time, ~5 min for 70B)
cargo run --release --bin rustyllm-shard -- /path/to/model

# Quantize per-layer F16 → Q4_K GGUF (~30 min for 70B)
./target/release/rustyllm-quantize-shards \
  --src /path/to/splits \
  --dst /path/to/splits_q4k \
  --format q4_k
```

Supported quant formats: `q4_0`, `q4_k`, `q5_k`, `q6_k`, `q8_0`, `q2_k`, `q3_k`.

## Backends

`cuda` and `metal` are mutually exclusive cargo features. `LoadOptions::default()` picks the best available at runtime: Metal → CUDA → CPU.

`parallel-cpu` (optional, opt-in) replaces candle's single-threaded CPU matmul with a rayon-parallel one in the F16 hybrid path. ~4× faster on 32-core hosts at our shapes. Works correctly alongside `cuda` (per-block dispatch picks the right kernel based on weight device).

`mkl` (optional, opt-in) links Intel oneMKL for F16 BLAS. **Doesn't improve perf on this stack** — the Q4 hybrid path uses `QMatMul` (ggml's Q4×Q8 integer kernel, already AVX-VNNI-tuned, bypasses BLAS), and the F16 hybrid is streaming-bound at residency configs small enough to avoid swap. Build with `--features cuda,mkl` only if you have a specific reason; needs Intel oneMKL 2021+ (`intel-oneapi-mkl-devel`) for the `hgemm_` symbol — Debian's stock `intel-mkl` package is too old.

## How it actually works

The `StreamingLlama` forward pass runs each transformer block in one of three tiers:

1. **GPU resident** (`pin_resident_layers(n)`): loaded once, kept on device, fastest per-token.
2. **CPU resident** (`pin_cpu_resident_layers(n)`): held in host RAM, compute on CPU via QMatMul or ParallelLinear, transferred GPU↔CPU at the tier boundary.
3. **Streamed**: weights mmap'd per token from disk shards. Slowest, but holds VRAM use bounded.

The hybrid mode is the unlock — instead of streaming all 80 layers from disk per token (Python AirLLM's model), keep as many layers as your GPU + RAM can hold. For a 70B Q4_K model on a 12 GB / 125 GB box, that's all 80 layers resident, zero per-token disk I/O.

Mixed-precision streaming adds an axis: the streamed tier (whatever doesn't fit elsewhere) loads quantized GGUF while the resident tiers stay F16. This works because PCIe bandwidth, not compute precision, is the streaming bottleneck.

Async H2D (`LoadOptions::async_h2d = true`, CUDA only) overlaps the next layer's PCIe transfer with the current layer's compute via double-buffered pinned host + ping-pong device buffers. Useful for pure streaming; incompatible with hybrid residency on small VRAM (the 3.5 GB pinned buffers don't fit alongside 4+ resident layers on 12 GB).

## Requirements

- Rust 1.70+
- CUDA toolkit (for `--features cuda`) or macOS 13+ (for `--features metal`); CPU fallback otherwise
- Disk space: ~140 GB for a 70B model in F16, ~38 GB in Q4_K

## License

Apache-2.0.
