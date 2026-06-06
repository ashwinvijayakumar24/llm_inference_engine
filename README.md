# From-Scratch LLM Inference Engine

A from-scratch inference engine for **Llama 3.2 1B Instruct** — implementing the transformer forward pass, KV cache, sampling, an OpenAI-compatible server, weight quantization, and a **custom CUDA attention kernel** that matches PyTorch's production `scaled_dot_product_attention`. Built in Python + CUDA C++.

> Anyone can call `model.generate()`. This project implements *what `generate()` does* — plus the serving-system optimizations around it.

**Hardware:** developed on Apple M4 (correctness), benchmarked on NVIDIA A100 40GB (GT PACE Phoenix).
**Validation:** every component diffed against HuggingFace `transformers` layer-by-layer.

---

## Highlights

- **Full transformer forward pass from scratch** — RoPE (with Llama-3 frequency scaling), grouped-query attention, RMSNorm, SwiGLU FFN, tied output projection — all hand-implemented in NumPy and validated against a HuggingFace oracle to **< 1e-3 logit error, exact greedy-token match**.
- **Custom CUDA decode-attention kernel** — built in 3 stages (serial → shared-memory reduction → split-KV flash-decoding with warp-shuffle reductions). The final kernel is **33× faster than the naive version** and **matches PyTorch SDPA (0.98–0.99×)** on A100, validated to < 1e-3 vs reference across 100+ random inputs.
- **Int8 / Int4 weight-only quantization** — int8 cuts weight memory **39%** for only **+0.14 perplexity**.
- **KV cache, sampling (greedy/temp/top-k/top-p), CLI, and an OpenAI-compatible streaming HTTP server.**
- **Benchmark harness** measuring TTFT, decode tok/s, p50/p99 inter-token latency, and memory — characterized against HuggingFace `transformers` and `llama.cpp` on identical hardware.

---

## Architecture

```
prompt
  → Tokenizer (HF tokenizers — library boundary)
  → Scheduler (run loop)
  → Model forward pass:
        embed lookup
        for each of 16 layers:
            RMSNorm → GQA attention (RoPE on q,k; read/write KV cache) → + residual
            RMSNorm → SwiGLU FFN → + residual
        final RMSNorm → LM head → logits
  → Sampler (greedy / temperature / top-k / top-p)
  → detokenize → stream out, loop until EOS / max_tokens
```

| Module | Role |
|--------|------|
| `engine/loader.py` | safetensors → tensors; fp32 (CPU reference), fp16 (GPU), int8/int4 (quantized) |
| `engine/components.py` | NumPy reference: RMSNorm, RoPE, GQA, SwiGLU |
| `engine/components_gpu.py` | PyTorch fp16 GPU versions + `linear()` quant chokepoint |
| `engine/model.py` / `model_gpu.py` | forward-pass wiring (CPU reference / GPU) |
| `engine/cache.py` | KV cache (NumPy + GPU fp16) |
| `engine/quant.py` | int8 per-channel + int4 group-wise quantization |
| `engine/sampler.py`, `scheduler.py` | sampling + generation loop |
| `engine/server.py`, `cli.py` | OpenAI-compatible HTTP + CLI |
| `kernels/attention_decode.cu` | custom CUDA decode-attention kernel (v1/v2/v3) |
| `bench/` | benchmark harness, baselines, perplexity eval |

### The "from-scratch" boundary

**Implemented from scratch:** every component's math (RoPE, GQA, RMSNorm, SwiGLU, attention, residual wiring), the KV cache, the scheduler, the quantization path, and the CUDA attention kernel.
**Library (deliberately):** array storage (NumPy/torch), the underlying GEMM (cuBLAS), the tokenizer (HF), safetensors parsing. Reimplementing BLAS is not the point — implementing the model and the serving system is.

---

## Benchmarks (NVIDIA A100 40GB)

### Engine vs. baselines (decode, fp16)

| Backend | Decode tok/s | Notes |
|---------|-------------|-------|
| This engine | ~79 | from-scratch reference |
| HuggingFace `transformers` | ~84 | fused kernels |
| `llama.cpp` (CUDA) | ~390 | mature, hand-optimized C++ |

*llama.cpp is ~5× faster — expected. The value of this project is the **relative deltas of its own optimizations** and how close a from-scratch engine gets to production, not beating llama.cpp.*

### Quantization (memory & quality)

| Mode | Weight memory | Δ memory | Perplexity | Δ perplexity |
|------|--------------|----------|-----------|-------------|
| fp16 | 2357 MB | — | 16.28 | — |
| int8 | 1430 MB | **−39%** | 16.42 | **+0.14** |
| int4 (g128) | 980 MB | −58% | 22.23 | +5.95 |

*int8 is near-free in quality. int4 at group-128 is too aggressive for a 1B model (small models are sensitive). Memory drop is below the theoretical 2×/4× because the 128k-vocab embedding/LM-head stays fp16; the quantized linear weights themselves drop exactly 2×/4×.*

### Custom CUDA decode-attention kernel (latency, µs)

| kv_seq | v1 (serial) | v2 (shared-mem) | v3 (split-KV) | PyTorch SDPA | v3 vs SDPA |
|--------|------------|----------------|--------------|--------------|-----------|
| 512 | 1569 | 365 | 189 | 185 | 0.98× |
| 1024 | 3120 | 714 | 189 | 185 | 0.98× |
| 2048 | 6225 | 1412 | 191 | 189 | 0.99× |

*v3 latency is **flat in sequence length** (split-KV parallelizes over the cache) — **33× faster than the naive v1 at kv_seq=2048** and matching PyTorch's optimized SDPA. End-to-end decode gains only ~4% because attention is not the bottleneck at these context lengths (the linear GEMMs dominate) — a deliberate, measured observation of where the time actually goes (Amdahl's law).*

---

## Quickstart

```bash
pip install -e .

# Download weights (gated — needs HF access to meta-llama/Llama-3.2-1B-Instruct)
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('meta-llama/Llama-3.2-1B-Instruct', local_dir='weights')"

# Generate (CLI, streaming)
llm-generate --prompt "Explain attention in one sentence." --max-tokens 80

# Serve (OpenAI-compatible)
uvicorn engine.server:app --port 8000
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":40,"stream":true}'
```

### GPU + CUDA kernel (NVIDIA, e.g. PACE A100)

```bash
module load cuda/12.9.1
bash scripts/build_kernels.sh                 # build the CUDA kernel module

python bench/harness.py --backend gpu --max-tokens 128                 # baseline
python bench/harness.py --backend gpu --cuda-attn v3 --max-tokens 128  # + custom kernel
python bench/harness.py --backend gpu --quant int8 --max-tokens 128    # + quantization
python -m bench.perplexity --mode int8                                 # quality eval
python -m bench.bench_attn_kernel                                      # kernel microbench
```

## Testing

```bash
pytest -m "not slow"                          # fast unit tests (CPU)
pytest tests/test_components_gpu.py -v         # GPU components (needs CUDA)
pytest tests/test_attention_kernel.py -v       # CUDA kernel correctness (100-input diff)
pytest -m slow -v                              # end-to-end identity checks (real weights)
```

Every optimization is validated against a correct reference **before** any speed measurement: the GPU path against the NumPy reference, the CUDA kernel against a torch reference (< 1e-3 across 100+ random inputs), and quantized/kernel decode against the unquantized/PyTorch path (identical greedy tokens).

---

## What I'd build next (future work)

- **Continuous batching** — a scheduler admitting/retiring requests mid-step, with a **batched-GEMM** step (stacked sequences) for true throughput scaling.
- **PagedAttention-style block KV cache** — eliminate per-sequence over-allocation; pairs with batching.
- **Fused int8/int4 GEMM** — recover the throughput that on-the-fly dequant currently costs (the quantization win here is memory, not speed).
- **Speculative decoding**, multi-GPU/tensor-parallel inference, fused prefill-attention kernel.

## Implementation log

`implemented.md` is a detailed per-phase build log: every component, why it was built, the concepts behind it, bugs hit and how they were fixed, and full benchmark numbers. Written as an interview-prep reference.
