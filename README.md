# freeToken-rs

A Rust port of [FreeToken](https://github.com/FlashML-org/FreeToken) — the edge-native
Mixture-of-Experts serving engine from *FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution* ([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)) —
built to measure how the engine's core ideas perform outside the PyTorch stack.

Expert weights live in CPU-cached pinned host RAM; the GPU holds a global LRU cache of
expert slots. Each decode step, routed experts that miss the cache are split between a
batched PCIe fetch and direct CPU compute by the paper's q\* bandwidth ratio, so both
finish together.

## Crates

| crate | what it is |
|---|---|
| `ft-core` | engine core: LRU `SlotCache`, q\* hybrid split policy, copy-run planner, q4_0 quant + AVX2 CPU GEMV (ggml `vec_dot_q4_0_q8_0` port) |
| `ft-gguf` | minimal GGUF v2/v3 reader (header, metadata, mmap'd tensors) |
| `ft-cuda` | CUDA kernels (build-time PTX, `compute_75`+): grouped q4_0 dequant-GEMV with shared-memory activations, silu/reduce, `HostBanks` (`cuMemHostRegister` pinned memory) |
| `ft-bench` | `ft-bench` (bandwidth calibration, port of `ft bench bw`) and `decode` (hybrid MoE decode benchmark, synthetic or real GGUF experts) |

## Results (RTX 4060 Ti 16 GB, Gemma-4-26B-A4B QAT q4_0)

**MoE expert-FFN kernel progression** (microbenchmark, ~95% cache-hit routing,
1862 GPU slots):

| kernel | ms/token | tok/s ceiling |
|---|---|---|
| naive per-expert launches | 20.9 | 48 |
| grouped launches (4/layer) | 19.6 | 51 |
| v2: smem activations, u16 loads, warp/row | 7.9 | 126 |
| **v3: dp4a integer dot over GPU-quantized q8 activations** | **~3** | **~300** |

**Full-model decode** (end-to-end generation, real routing, single stream):

| engine | tok/s | ms/token | notes |
|---|---|---|---|
| freeToken-rs, first working build | 16.0 | 63 | CPU attention, f32 kernels |
| freeToken-rs, profiled + optimized | 30.4 | 33 | dp4a kernels, rope tables, parallel router, fused MLP chain |
| freeToken-rs, + GPU attention & q4 lm_head | 42.2 | 24 | f16 KV cache on GPU, flash-decode kernel, overlapped MLP/MoE |
| freeToken-rs, GPU-resident decode | 46.8 | 21 | norms/rope/router/combine on device, pooled KV, 1 sync/layer |
| freeToken-rs, UVA experts + GPU sampling | 62.9 | 16 | misses stream over PCIe inside the kernel; 4-byte/token download |
| freeToken-rs, CUDA-graph replay | 64.8 | 15.4 | device LRU + topk; whole token = one graph launch |
| Python FreeToken, Triton fallback (driver 550) | 68.0 | 14.7 | CUDA graphs, Triton attention |
| freeToken-rs, two-phase expert fetch | 63.5 | 15.7 | misses cross PCIe once via copy stream; pair GEMVs read DRAM |
| freeToken-rs, flash-decode attention | 73.5 | 13.6 | position chunks fan out over SMs; online-softmax partials + merge |
| **freeToken-rs, fused qkv-prep (current)** | **76.1** | **13.1** | 8 head-op launches -> 1 kernel; fused gelu+q8 quantize |
| Python FreeToken, native accel (driver 580) | 91.8 | 10.9 | flashinfer + sglang-kernel |

Per-token attribution at 76 tok/s, measured by *subtractive graph profiling*
(re-capture the CUDA graph with one section stubbed out, difference the wall
times — per-kernel sync timers overstate whatever has the most launches):
moe pairs 3.6 · dense GEMVs 3.2 · lm_head 2.2 · attention 1.9 · everything
else ~2.2 ms. The MoE and dense sections sit near the card's ~288 GB/s
roofline (8 experts x 3.36 MB x 30 layers ≈ 807 MB/token for the pairs
alone); the residual gap to Python's native stack (~2 ms) is spread thin
across attention latency, dp4a GEMV efficiency, and launch overhead.

The two-phase fetch matters most when VRAM is short: at 1200 slots (31% of
the 3840 expert banks resident) decode improved 29.9 -> **46.6 tok/s**,
because a missed expert now crosses PCIe once by DMA instead of being read
in-kernel over UVA by both pair GEMVs while blocking SMs.

**GPU-poor hardware** (GTX 1650 4 GB laptop, driver 535): freeToken-rs runs the
same 26B model at **8.9 tok/s** in hybrid mode (280-slot cache, CPU experts +
PCIe streaming). Fetch-on-miss GPU routing manages only 5.2 tok/s there (7%
expert coverage, PCIe 3 — misses dominate), confirming FreeToken's
bandwidth-adaptive co-execution thesis on exactly the hardware it targets.
Python FreeToken cannot start on this machine at all — its torch 2.11 pin is
a CUDA-13 build requiring driver >= 580.

## Concurrency

`serve` implements continuous batching: up to `batch=N` sequences decode as
one batched forward per step (batched dense/expert/lm_head GEMVs via an
activation-indirection kernel; per-slot KV caches; prefill on admission;
finished sequences free their slot immediately). With the current kernels,
8 concurrent requests complete in **3.1 s wall** (was 5.5 s before the
two-phase/flash-decode round; 9.1 s serialized), 4 requests in 1.8 s, and a
single request in 0.9 s — batched output verified token-identical to
single-stream decoding (`btest`).

```
cargo test --release          # unit + GPU parity tests
cargo run --release -p ft-bench            # calibrate pcie/cpu bandwidth, q* fraction
cargo run --release -p ft-bench --bin decode -- gguf=model-q4_0.gguf slots=2048 \
    steps=200 warmup=100 fractions=0.0,0.2,1.0 locality=0.9
```

## Notes

- Ported from FreeToken (Apache-2.0); expert-cache semantics, q\* policy, and bank
  layouts follow `python/freetoken/moe/` in the reference implementation.
- Pinned host memory must be allocated with `cuMemHostRegister` on ordinary pages
  (`ft_cuda::HostBanks`), not `CU_MEMHOSTALLOC_WRITECOMBINED` — CPU reads from
  write-combined memory are ~100x slower and the hybrid path reads the same banks.
- In progress: full Gemma-4 forward pass (attention + tokenizer), AVX-512 CPU GEMV,
  OpenAI-compatible serving.
