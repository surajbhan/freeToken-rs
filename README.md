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
| freeToken-rs, + GPU attention & q4 lm_head | **42.2** | **24** | f16 KV cache on GPU, flash-decode kernel, overlapped MLP/MoE |
| Python FreeToken, Triton fallback (driver 550) | 68.0 | 14.7 | CUDA graphs, Triton attention |
| Python FreeToken, native accel (driver 580) | 91.8 | 10.9 | flashinfer + sglang-kernel |

Per-token profile at 42 tok/s (30 layers, 400-token context): moe 8.5 ·
lm_head 4.2 · router 3.6 · attn+o 3.7 · qkv 2.3 · shared-mlp 0.7 ·
rope/norms 0.9 ms. Known headroom: CUDA-graphing the decode step, GPU router,
fewer per-layer syncs, continuous batching. The Python engine's remaining
lead (native stack: 91.8 tok/s on the same GPU) is its CUDA graphs and fused
launch structure, not the language.

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
