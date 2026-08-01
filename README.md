# From-Scratch LLM Inference Engine

A from-scratch inference engine for **Llama 3.2 1B Instruct** — implementing the transformer forward pass, KV cache, sampling, an OpenAI-compatible server, weight quantization, and a **custom CUDA attention kernel** that matches PyTorch's production `scaled_dot_product_attention`. Built in Python + CUDA C++.

> Anyone can call `model.generate()`. This project implements *what `generate()` does* — plus the serving-system optimizations around it.

**Hardware:** developed on Apple M4 (correctness), benchmarked on NVIDIA A100 40GB (GT PACE Phoenix).
**Validation:** every component diffed against HuggingFace `transformers` layer-by-layer.

---

## Highlights

- **Full transformer forward pass from scratch** — RoPE (with Llama-3 frequency scaling), grouped-query attention, RMSNorm, SwiGLU FFN, tied output projection — all hand-implemented in NumPy and validated against a HuggingFace oracle to **< 1e-3 logit error, exact greedy-token match**.
- **Custom CUDA decode-attention kernel** — built in 3 stages (serial → shared-memory reduction → split-KV flash-decoding with warp-shuffle reductions). The final kernel **matches PyTorch SDPA (0.98–0.99×)** at 512–2048 KV length on A100, validated to < 1e-3 vs reference across 100+ random inputs.
- **Int8 / Int4 weight-only quantization** — int8 cuts weight memory **39%** (2357 → 1430 MB) for **+0.04 WikiText-2 perplexity**.
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
| `engine/sampler.py`, `scheduler.py` | sampling + generation loop (single request) |
| `engine/server.py`, `cli.py` | OpenAI-compatible HTTP + CLI (single-request reference path) |
| `kernels/attention_decode.cu` | custom CUDA decode-attention kernel (v1/v2/v3) |
| `kernels/bindings.cpp` | nanobind module — passes device pointers, no host round-trip |
| `bench/` | benchmark harness, HF + llama.cpp baselines, perplexity eval |
| `bench/results/` | committed benchmark artifacts (CSV/JSON) behind every number below |
| `tests/` | oracle-based correctness suite (CPU, GPU, CUDA kernel) |

### Public API

```python
from engine import LlamaModelGPU, generate, get_sampler, load_config, load_weights_gpu

config  = load_config("weights")
model   = LlamaModelGPU(load_weights_gpu("weights", config), config)

for token_id in generate(model, prompt_ids, get_sampler(temp=0.0), max_tokens=64):
    ...
```

`engine/__init__.py` declares the supported surface in `__all__`; everything else is internal. Imports are lazy, so `import engine` does not pull in torch or CUDA.

### The "from-scratch" boundary

**Implemented from scratch:** every component's math (RoPE, GQA, RMSNorm, SwiGLU, attention, residual wiring), the KV cache, the scheduler, the quantization path, and the CUDA attention kernel.
**Library (deliberately):** array storage (NumPy/torch), the underlying GEMM (cuBLAS), the tokenizer (HF), safetensors parsing. Reimplementing BLAS is not the point — implementing the model and the serving system is.

---

## Benchmarks (NVIDIA A100 40GB)

Summary below. **[`BENCHMARKS.md`](BENCHMARKS.md) is the source of record** — full methodology (iteration counts, warmup, timing method, batch size), per-prompt tables, reproduction commands, and a stated list of known gaps.

All numbers are **batch 1**, greedy decoding, single request. Nothing here describes behaviour under concurrent load.

### Engine vs. baselines (decode, fp16)

| Backend | Decode tok/s | Notes |
|---------|-------------|-------|
| This engine | ~79 | from-scratch reference |
| HuggingFace `transformers` | ~84 | fused kernels |
| `llama.cpp` (CUDA) | ~390 | mature, hand-optimized C++ |

*llama.cpp is ~5× faster — expected. The value of this project is the **relative deltas of its own optimizations** and how close a from-scratch engine gets to production, not beating llama.cpp.*

*Method: 128 max new tokens, 3 prompts, 1 warmup + 3 measured runs, host-clock timing (both sides sync every step). The HF baseline drives a hand-rolled `past_key_values` decode loop rather than `model.generate()`, so it mirrors this engine's loop instead of measuring `generate()` overhead. [Details →](BENCHMARKS.md#benchmark-1--engine-vs-baselines-decode-throughput-fp16)*

### Quantization (memory & quality)

| Mode | Weight memory | Δ memory | WikiText-2 ppl | Δ ppl |
|------|--------------|----------|-----------|-------------|
| fp16 | 2357 MB | — | 14.37 | — |
| int8 | 1430 MB | **−39%** | 14.41 | **+0.04** |
| int4 (g128) | 980 MB | −58% | 18.82 | +4.45 |

*Perplexity on the WikiText-2 raw test split, sliding window 512 / stride 256, first 100k tokens. fp16 at 14.37 sits in the expected range for Llama-3.2-1B — an external cross-check on the whole forward pass.*

*int8 is near-free in quality. int4 at group-128 is too aggressive for a 1B model (small models are sensitive). Memory drop is below the theoretical 2×/4× because the 128k-vocab embedding/LM-head stays fp16; the quantized linear weights themselves drop exactly 2×/4×. Note that quantization here **costs** throughput (~79 → ~45 → ~22 tok/s) — weights are dequantized on the fly, so the win is memory, not speed.*

*An earlier run on a short synthetic text gave +0.14 for int8; the WikiText-2 measurement supersedes it. Both agree on the conclusions — see [BENCHMARKS.md](BENCHMARKS.md#benchmark-4--quantization-memory-and-quality) for what changed and why.*

### Custom CUDA decode-attention kernel (latency, µs)

| kv_seq | v1 (serial) | v2 (shared-mem) | v3 (split-KV) | PyTorch SDPA | v3 vs SDPA |
|--------|------------|----------------|--------------|--------------|-----------|
| 128 | 726 | 176 | 300 | 184 | **0.61×** |
| 512 | 1569 | 365 | 189 | 185 | 0.98× |
| 1024 | 3120 | 714 | 189 | 185 | 0.98× |
| 2048 | 6225 | 1412 | 191 | 189 | 0.99× |

*v3 latency is **flat in sequence length** (split-KV parallelizes over the cache) and **matches PyTorch's optimized SDPA at 512–2048**. At kv_seq=128 it loses (0.61×) — two kernel launches plus a scratch allocation aren't amortized when there's little work.*

*The "33× over v1" figure (6225 → 191 µs at kv_seq=2048) compares against a **deliberately serial baseline**: v1 launches `<<<32, 1>>>` — 32 blocks of one thread each, the CPU reference transcribed to a single CUDA thread as the first rung of the learning progression. It measures parallelism left on the table by construction, not a win over a real implementation. **The meaningful result is matching SDPA.***

*End-to-end decode gains only ~4% (59.5 → 62.0 tok/s, same node). Two causes: attention is a minor fraction of a decode step at these lengths — the linear GEMMs dominate (Amdahl) — **and** the kernel's layout requirement forces a full KV-cache transpose per layer per token that the PyTorch path doesn't pay. [Both quantified →](BENCHMARKS.md#benchmark-3--kernel-end-to-end-amdahl)*

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

> The HTTP server is a **single-request reference path** — it demonstrates the OpenAI protocol surface and SSE streaming, and currently runs the CPU reference model. It has no queue, no batching, and no request isolation: concurrent requests serialize. Multi-request serving is deliberately out of scope here (see future work).

### GPU + CUDA kernel (NVIDIA, e.g. PACE A100)

```bash
module load cuda/12.9.1
bash scripts/build_kernels.sh                 # build the CUDA kernel module

python bench/harness.py --backend gpu --max-tokens 128                 # baseline
python bench/harness.py --backend gpu --cuda-attn v3 --max-tokens 128  # + custom kernel
python bench/harness.py --backend gpu --quant int8 --max-tokens 128    # + quantization
python -m bench.baseline_hf --max-tokens 128 --attn-impl sdpa          # HF baseline
python -m bench.bench_attn_kernel                                      # kernel microbench

# Quality eval on a standard corpus (sliding window, WikiText-2 protocol)
pip install -e '.[bench]'
python scripts/fetch_wikitext2.py
python -m bench.perplexity --mode int8 --text bench/wikitext2_test.txt \
    --window 512 --stride 256 --max-tokens 100000
```

Results are written to `bench/results/` as JSON + CSV, stamped with hostname, GPU, and library versions.

## Testing

```bash
pytest -m "not slow"                          # fast unit tests (CPU)
pytest tests/test_components_gpu.py -v         # GPU components (needs CUDA)
pytest tests/test_attention_kernel.py -v       # CUDA kernel correctness (100-input diff)
pytest -m slow -v                              # end-to-end identity checks (real weights)
```

Every optimization is validated against a correct reference **before** any speed measurement: the GPU path against the NumPy reference, the CUDA kernel against a torch reference (< 1e-3 across 100+ random inputs), and quantized/kernel decode against the unquantized/PyTorch path (identical greedy tokens).

---

## Serving-layer seam

The engine exposes a pluggable attention/KV backend so a serving layer can supply a **paged, batched** implementation without forking this repo:

```python
from engine import AttentionBackend, BatchMeta   # importing these does not pull in torch

class MyPagedBackend:                       # structural — no base class to inherit
    def append_kv(self, layer_idx, k, v, meta: BatchMeta) -> None: ...
    def attend(self, q, layer_idx, scale, meta: BatchMeta): ...

logits = model.forward_varlen(token_ids, meta, backend)   # (n_seqs, vocab), on device
```

When a backend is supplied it owns **both** the KV write and the attention math, and the engine stops touching the KV cache entirely. Sequences are packed along one token axis (variable-length, not padded), which is what keeps every linear, norm, and FFN in the forward pass unchanged.

Three things worth stating plainly:

- **`prefill()` and `decode_step()` are unmodified.** `forward_varlen()` is a sibling, so every benchmark and correctness claim above still describes live code.
- **The custom CUDA kernel is not used on a paged path.** Its nanobind ABI takes no strides, block table, or block size (`kernels/bindings.cpp`), and `Q` has no batch dimension — a paged batched kernel would be a rewrite, not an extension. The kernel remains the single-request contiguous decode path.
- **Continuous batching, scheduling, block allocation, and eviction are deliberately *not* here.** This repo exposes the seam; the serving layer owns the policy.

See [`engine/attention_backend.py`](engine/attention_backend.py) for the full contract.

## What I'd build next (future work)

- **Fused batched attention on the paged path** — the seam exists; a backend that batches efficiently is the remaining work.
- **PagedAttention-style block KV cache** — eliminate per-sequence over-allocation. Now implementable behind `AttentionBackend` without engine changes.
- **Fused int8/int4 GEMM** — recover the throughput that on-the-fly dequant currently costs (the quantization win here is memory, not speed).
- **Speculative decoding**, multi-GPU/tensor-parallel inference, fused prefill-attention kernel.

## Going deeper

- **[`BENCHMARKS.md`](BENCHMARKS.md)** — every measured number with its methodology, reproduction commands, and a stated list of known gaps.
- **[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md)** — per-phase build log: every component, why it was built, the concepts behind it, the bugs hit and how they were fixed, and the original run output.

## License

MIT — see [LICENSE](LICENSE). Model weights are not included and are subject to Meta's Llama 3.2 license.
