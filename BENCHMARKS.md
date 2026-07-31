# Benchmarks

Every measured number in this project, with the methodology and the command that produces it.
Source of record for all figures quoted in `README.md`.

**Hardware:** NVIDIA A100 40GB (Georgia Tech PACE Phoenix, interactive partition), CUDA 12.9.1, fp16 unless stated.
Correctness development on Apple M4 (CPU/NumPy); all performance numbers on A100.

**Artifacts:** every figure below is backed by a committed file in `bench/results/` — CSV and JSON, stamped with hostname, GPU, and library versions. `notes/implemented.md` is the per-phase build log with the original reasoning and the bugs hit along the way.

---

## Reading these numbers honestly

Two caveats that apply to everything below. They are here rather than in a footnote because they change how the numbers should be read.

**1. Cross-session absolute throughput is not comparable.** PACE A100 nodes vary in contention and clock behaviour. The Phase 3 baselines (~79 tok/s) ran on node `-33-0`; the kernel off/on pair (59.5 → 62.0 tok/s) ran on `-31-0`. The node ID is in every result filename, so this is checkable rather than asserted. Only same-node comparisons are used, and each table marks which comparison it supports.

**2. Everything is batch 1, single request.** No claim here describes behaviour under concurrent load.

---

## Benchmark 1 — Engine vs. baselines (decode throughput, fp16)

**Question:** how close does a from-scratch engine get to a mature one on identical hardware?

| Backend | Decode tok/s | p99 ITL | Notes |
|---------|-------------|---------|-------|
| **This engine** | ~79 | ~13 ms | from-scratch reference |
| HuggingFace `transformers` | ~84 | ~12 ms | **~6% faster** |
| `llama.cpp` (CUDA) | ~390 | ~2.6 ms | mature hand-optimized C++ (~5×) |

Per-prompt detail:

| Prompt | Tokens | This engine | HF | llama.cpp |
|--------|--------|------------|-----|-----------|
| short  | 40 | ~79 tok/s (p50 12.6 / p99 13.2 ms) | ~83 tok/s (p99 12.8 ms) | ~393 tok/s |
| medium | 46 | ~79 tok/s (p50 12.6 / p99 12.8 ms) | ~84 tok/s (p99 12.1 ms) | ~390 tok/s |
| long   | 57 | ~80 tok/s (p50 12.5 / p99 12.8 ms) | ~84 tok/s (p99 12.0 ms) | ~395 tok/s |

TTFT was ~0.01 s for all three prompts — the prompts are short, so prefill is not the interesting quantity here.

### Confirmation run, with the attention implementation pinned

The June 5 run left `attn_implementation` to the library default and recorded neither it nor the `transformers` version. A later run on node `-11-0` pins it explicitly and verifies what actually loaded (`Active attention implementation: sdpa`, transformers 5.10.2 / torch 2.12.0+cu130):

| Backend | Decode tok/s | p99 ITL | Node |
|---------|-------------|---------|------|
| This engine | 60.8 | 16.5 ms | `-11-0` |
| HuggingFace (SDPA, pinned) | 63.8 | 15.7 ms | `-11-0` |
| **Gap** | **4.8%** | | |

Absolute throughput is far below the `-33-0` numbers (60.8 vs ~79) — the same node-contention effect flagged above, and a good illustration of why only same-node comparisons are used. **The relative gap is the stable quantity: 5.6% on `-33-0`, 4.8% on `-11-0`.** The "within 6%" claim now holds across two independent nodes, once with the attention implementation confirmed rather than assumed.

### Methodology

- **Batch size 1.** The engine has no batch dimension; this is a single-request latency benchmark, not a throughput-under-load benchmark.
- **Greedy decoding**, 128 max new tokens, 3 prompts (40/46/57 tokens after the chat template), 1 warmup run discarded, 3 measured runs per prompt.
- **`decode_tok_s` excludes prefill** — it is `(n_tokens − 1) / Σ(inter-token latencies)`, so TTFT does not contaminate it (`bench/harness.py:111`).
- **Timing:** `time.perf_counter()` on the host around each yielded token. Valid here because both sides force a device synchronise every step — this engine at `engine/model_gpu.py:158` (`.cpu().float().numpy()`), HF at `bench/baseline_hf.py:67` (`int(...argmax())`). Neither side can hide work behind an async queue, so the host clock misses nothing on either.
- **The HF baseline is deliberately not `model.generate()`.** It hand-rolls the decode loop with `past_key_values` and `argmax` (`bench/baseline_hf.py:61-71`) so it mirrors this engine's loop structure. Timing `generate()` would measure its wrapper overhead and flatter this engine unfairly.
- **HF attention implementation:** pinned to `sdpa` via `--attn-impl` (`bench/baseline_hf.py`); the loaded implementation is read back after construction and written into the results row. The June 5 run predates this and recorded neither — closed by the confirmation run above.
- **llama.cpp** uses `llama-bench` with synthetic prompts of matching token lengths (40/46/57) and fp16 GGUF weights. tok/s depends on sequence length rather than content, so this is a fair throughput comparison even though the prompt text differs.

### Reproduce

```bash
python bench/harness.py     --backend gpu --max-tokens 128 --n-runs 3
python -m bench.baseline_hf --max-tokens 128 --n-runs 3 --attn-impl sdpa
python bench/baseline_llamacpp.py
```

### What this shows

The ~6% gap to HF is the honest headline: a from-scratch engine calling separate matmuls, with RoPE / softmax / RMSNorm as distinct ops, lands within noise-adjacent distance of a library using fused kernels. The 5× gap to llama.cpp is expected and is not the point — llama.cpp has custom fused CUDA kernels, quantization-aware GEMM, and years of tuning. The value here is the **relative deltas of this engine's own optimizations**, each measured against its own baseline.

---

## Benchmark 2 — Custom CUDA decode-attention kernel (latency)

**Question:** can a hand-written decode-attention kernel match PyTorch's production SDPA?

Per-call latency in microseconds, batch 1, 32 query heads / 8 KV heads / head_dim 64 — the real Llama-3.2-1B decode shape.

| kv_seq | v1 (serial) | v2 (shared-mem) | v3 (split-KV) | PyTorch SDPA | **v3 vs SDPA** |
|--------|------------|----------------|--------------|--------------|-----------|
| 128  | 726  | 176  | 300 | 184 | **0.61×** *(v3 loses)* |
| 512  | 1569 | 365  | 189 | 185 | 0.98× |
| 1024 | 3120 | 714  | 189 | 185 | 0.98× |
| 2048 | 6225 | 1412 | 191 | 189 | 0.99× |

**v3 is flat in sequence length** — 189 / 189 / 191 µs across 512 → 2048 — because split-KV parallelizes over the KV cache itself. Flatness is the evidence that the occupancy fix landed.

**At kv_seq=128, v3 is 39% slower than SDPA.** Two kernel launches (partial + combine) plus a `cudaMalloc`/`cudaFree` pair for the scratch buffers are not amortized when there is little work to do. This row is reported because omitting it would make the kernel look uniformly competitive when it is not.

### What "33× over naive" means, precisely

`6225 / 191 = 32.6×` at kv_seq=2048, comparing **v3 against v1**.

**v1 is not a naive-but-reasonable kernel. It launches `<<<n_heads, 1>>>` — 32 blocks of one thread each** (`kernels/attention_decode.cu:91`), occupying ~32 of an A100's 6912 CUDA cores. It is the CPU reference transcribed to a single CUDA thread, written deliberately as the first rung of a three-stage learning progression. The 33× measures how much parallelism was left on the table by construction, not a win over a real implementation.

**The defensible result is v3 ≈ 0.98–0.99× PyTorch SDPA at realistic context lengths.** Matching the vendor-optimized kernel is the harder and more meaningful claim.

### The three stages

| Version | Structure | What it taught |
|---|---|---|
| v1 | 1 block/head, 1 thread; streaming (online) softmax | Flash-Attention's core insight in serial form — O(head_dim) state, not O(kv_seq) |
| v2 | 1 block/head, head_dim threads; shared-memory tree reduction | `__shared__`, `__syncthreads()`, parallel reduction. 4–8× over v1, but still scales with kv_seq — only 32 blocks on 108 SMs |
| v3 | (n_heads × n_splits) blocks; warp-shuffle reduction + flash-decoding combine | Occupancy is the lever. `__shfl_down_sync` in registers, no shared memory inside a warp |

The v2 → v3 win came from **launching more blocks**, not from a tighter inner loop. That is the transferable lesson.

### Methodology

- **CUDA events** around a 200-iteration loop, after 20 warmup iterations, divided by the count (`bench/bench_attn_kernel.py:28-44`). Events record on the GPU stream, so this is GPU time, not host time.
- **Mean, not median or p99** — a single outlier would inflate it.
- Inputs `randn * 0.1`, fp16 in/out with fp32 accumulation, matching the kernel's precision policy.
- **The custom kernels sync after every call** (`kernels/bindings.cpp:67,79,91`) while SDPA queues asynchronously — this biases the comparison *against* the custom kernel. The numbers are conservative.
- **Known unfixed inefficiency:** the v3 launcher calls `cudaMalloc`/`cudaFree` for its scratch buffers on every invocation (`kernels/attention_decode.cu:297-300, 311`), inside the timed path. A persistent buffer sized at construction would remove it. Left in place because the end-to-end measurement below showed attention was not the bottleneck.

### Reproduce

```bash
bash scripts/build_kernels.sh          # needs CUDA 12.x, sm_80
python -m bench.bench_attn_kernel      # -> bench/results/attn_kernel_microbench.csv
```

---

## Benchmark 3 — Kernel end-to-end (Amdahl)

**Question:** what does a 33× faster attention kernel buy in real generation?

Same A100 node, back-to-back, kernel off vs on — **this is a valid same-session comparison.**

| Config | Decode tok/s | p50 ITL |
|--------|-------------|---------|
| kernel off (PyTorch attention) | 59.5 | 16.8 ms |
| kernel on (v3) | 62.0 | 16.1 ms |
| **Delta** | **+4.2%** | −0.7 ms |

*(These absolute values are lower than Benchmark 1's ~79 tok/s because they come from a different, more contended node. The +4.2% delta is the meaningful figure; the absolutes are not comparable across sessions.)*

### Why only 4%, completely

**Cause 1 — Amdahl.** At these context lengths attention is a minor fraction of a decode step. The step is dominated by the linear GEMMs: Q/K/V/O projections plus the 2048 → 8192 → 2048 FFN, roughly 60.8 M parameters of weight read *per layer*, 16 layers deep. Attention at kv_seq=2048 reads ~4.2 MB of KV cache by comparison. Decode at batch 1 is weight-bandwidth-bound. Speeding up ~15% of the work caps the gain near 15%.

**Cause 2 — the kernel's layout requirement adds a copy the baseline does not pay.** The KV cache is stored `(kv_seq, n_kv_heads, head_dim)` (`engine/cache.py:28`); the kernel requires `(n_kv_heads, kv_seq, head_dim)` (`kernels/attention_decode.cu:31`). So `engine/components_gpu.py:153-154` calls `.transpose(0,1).contiguous()` on the **full K and V cache, every layer, every decode step** — roughly 67 MB of copy traffic per token at kv_seq=2048 that the PyTorch path never pays. Part of the attention win is spent back on memcpy.

The split between the two causes has **not** been measured; isolating it would require storing the cache in kernel layout. The fix for cause 2 is cheap (store transposed, or teach the kernel a stride) and is the natural next step if this were pursued.

---

## Benchmark 4 — Quantization (memory and quality)

**Question:** how much weight memory does int8/int4 save, and what does it cost in quality?

| Mode | Weight memory | Δ memory | WikiText-2 ppl | Δ ppl | Decode tok/s |
|------|--------------|----------|-----------|-------------|-------------|
| fp16 | 2357.1 MB | — | 14.3695 | — | ~79 |
| **int8** (per-channel) | 1429.8 MB | **−39%** | 14.4134 | **+0.044** | ~45 |
| int4 (group-128) | 979.6 MB | −58% | 18.8194 | +4.450 | ~22 |

Perplexity measured on the **WikiText-2 raw test split**, sliding window 512 / stride 256, first 100,000 tokens, A100, transformers 5.10.2 / torch 2.12.0+cu130. Artifact: `bench/results/perplexity.csv`.

fp16 at **14.37** is in the expected range for Llama-3.2-1B on WikiText-2 — that is the external cross-check. It corroborates the whole forward pass (tokenizer, RoPE scaling, GQA, the layer stack) against something outside this repo, which no other measurement here does.

**All three values replicated bit-identically on a second, different A100 node** (`-31-0` and `-11-0`; both runs are in `perplexity.csv`). Perplexity is a deterministic function of the weights and the eval protocol, so this is the expected result — but it confirms there is no hidden nondeterminism in the forward pass, and it is the one number here that node contention cannot move.

### Memory — verified independently

These recompute exactly from `weights/config.json` (hidden 2048, 16 layers, FFN 8192, vocab 128256, `tie_word_embeddings: true`):

- Quantized linears — the 7 per-layer projections only (`engine/loader.py:111-119`): **973 M params**
- Embedding, tied to `lm_head` (`engine/loader.py:99-100`), deliberately left fp16: **262.7 M params**
- fp16: (973 + 262.7) M × 2 B = **2357 MB** ✓
- int8: 973 M × 1 B + 262.7 M × 2 B = **1429 MB** ✓ (−39.3%)
- int4-g128, including fp16 group scales: **980 MB** ✓ (−58%)

Measured rather than assumed: `bench/harness.py:54-66` sums actual stored bytes and de-duplicates the tied `lm_head`/`embed_tokens` alias by `id()`. Without that de-duplication fp16 would be overstated by 525 MB and the compression ratio would look better than it is.

**Why not the naive 2×/4×?** The 128k-vocab embedding stays fp16 and is 525 MB of the int8 total — disproportionately large for a 1B model. The *quantized linears themselves* drop exactly 2× and 4×. On a 7B+ model the linears dominate and the reduction approaches the theoretical limit.

### Throughput drops, and that is expected

int8 and int4 are **slower**, not faster (~79 → ~45 → ~22 tok/s). Weights are dequantized on the fly (`engine/components_gpu.py:22-24`): reconstruct the fp16 tile, then GEMM — and int4 also unpacks nibbles. The win in this phase is **memory, not speed**. Recovering the speed requires a fused int8/int4 GEMM that multiplies directly in low precision; deliberately out of scope. The quantization *math* is what was validated here.

### The earlier synthetic-text measurement, and what changed

An earlier run measured perplexity on `bench/computing_history.txt` — ~450 words of original prose, single window, no stride — in a file then misnamed `wikitext_sample.txt`. It gave fp16 16.28 / int8 16.42 / int4 22.23, i.e. **+0.14** and **+5.95**.

Both measurements agree on every conclusion: int8 is near-free, int4-g128 is unusable at 1B. But two things moved, and the direction of one is worth recording because it contradicts the reasoning used to justify the rerun:

- **The absolute perplexity fell** (16.28 → 14.37). WikiText-2 is *more* predictable for this model than the synthetic paragraph — encyclopedic prose is closer to its training distribution.
- **The int8 penalty shrank by 3×** (+0.14 → +0.044). The prediction going in was that smooth, low-entropy synthetic text would *understate* quantization damage, since quantization noise is least likely to flip a token the model is already confident about. The opposite happened: the real corpus shows *less* damage. The synthetic number was pessimistic, not optimistic — most likely because a ~600-token sample gives a noisy delta, not because of any property of the text.

The lesson is about sample size rather than text difficulty. Quote **+0.04 on WikiText-2**, and treat the earlier number as superseded.

### Reproduce

```bash
pip install -e '.[bench]'
python scripts/fetch_wikitext2.py                       # login node — compute nodes may lack internet
for m in fp16 int8 int4; do
  python -m bench.perplexity --mode $m --text bench/wikitext2_test.txt \
      --window 512 --stride 256 --max-tokens 100000
done                                                    # -> bench/results/perplexity.csv
```

### Why int4 fails at 1B and int8 does not

Three compounding effects. **Levels:** int8 symmetric gives 255 levels, int4 gives 15 — about 17× coarser. **Outliers:** both use symmetric absmax scaling, so one large weight sets the scale for its whole row or group; with 7 positive levels a single outlier can push most of a group into 1–2 levels, while int8 has levels to spare. **Model size:** a 1B model has far less redundancy than a 70B, so per-weight error propagates instead of averaging out. This is why int4 is routine at 70B and destructive here. A finer group size (32) would narrow the gap at ~12% scale overhead instead of ~3%; untested.

---

## Correctness gates (measured before any speed number)

Every optimization was validated against a correct reference *before* it was timed.

| Check | Result |
|-------|--------|
| Final logits vs fp32 HuggingFace oracle (2 prompts, all positions) | **max-abs diff < 1e-3** |
| Argmax vs oracle, every position | **exact match, zero mismatches** |
| 32 greedy tokens vs HF `generate(do_sample=False)` | **bit-identical** |
| KV-cache generation vs no-cache generation | **identical tokens** |
| CUDA kernel vs torch reference, 100 random inputs, kv_seq 1–512 | **max-abs diff < 1e-3, all versions** |
| CUDA kernel end-to-end, on vs off, real weights | **identical greedy tokens** |
| GQA routing (constructed inputs, distinct value per KV head) | **each query head reads head // 4** |
| int8 model first-token argmax vs fp16 | **match** |

Tolerances are max-absolute, not mean. GPU-vs-CPU component tests use `atol=1e-2`, which is appropriate for fp16 — its relative epsilon is ~1e-3, so on activations of magnitude ~10 the representational error alone is ~1e-2. The sensitive operations deliberately do not run in fp16: RMSNorm computes in fp32 (`engine/components_gpu.py:28-31`), attention scores are fp32 before softmax (`:169`), RoPE tables are fp32, and the CUDA kernel accumulates in fp32 with fp16 only at the boundaries.

```bash
python3 tests/oracle.py                        # regenerate HF fixtures
pytest -m "not slow"                           # fast CPU tests
pytest tests/test_attention_kernel.py -v       # kernel correctness (needs CUDA)
pytest -m slow -v                              # end-to-end identity, real weights
```

---

## Known gaps

Stated here rather than discovered by a reader.

1. **`bench/harness.py` reports host RSS as `peak_mem_mb` while `bench/baseline_hf.py` reports `torch.cuda.max_memory_allocated()`** — same column name, different quantity. Do not compare those two rows. The weight-memory figures in Benchmark 4 are a separate, reliable measurement.
2. **Everything here is batch 1.** There is no batch dimension in the tensors, the KV cache, the causal mask, or the CUDA kernel. No claim in this document describes behaviour under concurrent load.
3. **The kernel end-to-end split is unmeasured.** Benchmark 3 names two causes for the ~4% gain but does not quantify their relative contribution.
