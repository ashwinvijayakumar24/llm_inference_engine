# implemented.md — Build Log & Learning Reference

This document tracks every component built in the LLM inference engine — what it does, why it was built, and the concepts behind it. Updated after each phase. Intended as a reference for interviews and for understanding the project end-to-end.

---

## Phase 0 — Setup

### 0.1 — Repo + Python Toolchain ✅

**Completed:** 2026-06-04

#### What was built

| Path | Purpose |
|------|---------|
| `engine/` | All Python inference engine code |
| `engine/loader.py` | Weight loading from safetensors files |
| `engine/model.py` | Forward pass wiring for the full transformer |
| `engine/components.py` | From-scratch math primitives (RMSNorm, RoPE, GQA, SwiGLU) |
| `engine/cache.py` | KV cache (naive contiguous first, paged later) |
| `engine/scheduler.py` | Generation loop and request scheduling |
| `engine/sampler.py` | Token sampling strategies |
| `engine/server.py` | FastAPI OpenAI-compatible HTTP endpoint |
| `engine/cli.py` | Command-line interface for generation |
| `kernels/` | CUDA C++ kernel source files (Phase 4) |
| `tests/` | Test suite — unit tests and oracle comparisons |
| `bench/` | Benchmark harness and result CSVs |
| `scripts/` | One-off utility scripts (weight inspection, tokenizer checks) |
| `notes/` | Config values, tensor shape dumps, model hyperparameters |
| `pyproject.toml` | Package definition and dependency list |
| `.gitignore` | Excludes weights, build artifacts, and `__pycache__` |

#### Why this structure

The directory layout mirrors the PRD §5 specification. Each concern is isolated: the math lives in `components.py`, the wiring in `model.py`, the serving in `server.py`. This makes each file independently testable — you can diff a single component against a HuggingFace oracle without running the full engine.

Stub modules (all currently `raise NotImplementedError`) exist so the import graph is valid from day one. Any file that tries to `import engine.model` will succeed — it just won't run until implemented. This prevents phantom import errors from masking real bugs.

#### Key dependencies and why each was chosen

| Package | Role | Why, not an alternative |
|---------|------|------------------------|
| `numpy` | Reference math for all Phase 1–3 components | Explicit array operations make every formula readable and debuggable. No autograd overhead, no framework magic. |
| `safetensors` | Load model weights | Memory-mapped, safe (no pickle exploit surface), standard for modern HF models. |
| `tokenizers` | Tokenize prompts | Rust-backed Hugging Face library — fast, correct, handles Llama's BPE + chat template. Per PRD: tokenizer is a library boundary, not from-scratch. |
| `transformers` | HuggingFace oracle for correctness validation | The reference implementation we diff against in Phase 1. Not used in the engine's hot path. |
| `torch` | GPU tensor operations (Phase 3+) and oracle forward passes | Needed for the HF oracle and for the PACE GPU path. On Mac: CPU/MPS only. |
| `fastapi` + `uvicorn` | OpenAI-compatible HTTP serving | FastAPI gives typed request/response models and automatic `/docs`. Uvicorn is the ASGI server behind it. |
| `huggingface_hub` | Downloading model weights | Standard HF snapshot download; handles sharded safetensors and auth tokens. |
| `pytest` | Test runner | Standard Python test framework. Every acceptance criterion in TASKS.md maps to a test. |

#### Concepts to know

**Why `pip install -e .` (editable mode)?**  
Installs the package as a live symlink into site-packages. Changes to `engine/` take effect immediately without reinstalling. Essential during active development.

**Why stub modules rather than empty files?**  
Stubs with `raise NotImplementedError` give a clear failure mode: if you accidentally call an unimplemented function, you get an explicit error pointing at the function name, not a confusing `AttributeError` or silent wrong behavior.

**What is `pyproject.toml`?**  
The modern Python package spec (PEP 517/518). Replaces `setup.py`. Defines: package name, version, Python version requirement, runtime dependencies, optional dev dependencies, and entry points (the `llm-generate` CLI command). The `[tool.pytest.ini_options]` section tells pytest where to look for tests.

---

## Phase 0.2 — Weight Download + Inspection ✅

**Completed:** 2026-06-04

#### What was built

| Path | Purpose |
|------|---------|
| `MODEL.md` | Records chosen model, HF repo, and gate status |
| `weights/` | Downloaded model files (gitignored — ~2.5 GB) |
| `scripts/inspect_weights.py` | Opens safetensors, dumps every tensor name/shape/dtype, asserts all shapes against config |
| `notes/tensor_dump.txt` | Full tensor name/shape/dtype output (146 tensors, committed) |
| `notes/model_config.md` | All hyperparameters, derived shapes, RoPE scaling details |

#### Key discoveries

**146 tensors total.** 16 layers × 9 tensors/layer (q/k/v/o, gate/up/down, input_norm, post_attn_norm) + embed + final_norm = 146. No `lm_head.weight` — confirmed tied to `model.embed_tokens.weight`.

**All weights are bfloat16.** NumPy does not support bfloat16 — the safetensors `framework="numpy"` backend fails. Use `framework="pt"` (torch) for loading. In Phase 1 reference code, cast to float32 after loading.

**Llama 3 scaled RoPE ⚠️.** This is NOT standard RoPE. The model uses `rope_type="llama3"` with a frequency scaling factor of 32× for low-frequency components. Plain `rope_theta=500000` without the scaling would produce wrong positional encodings. See `notes/model_config.md` for full spec.

#### Architecture numbers to memorize

```
hidden_size:            2048
num_attention_heads:    32    (query heads)
num_key_value_heads:    8     (KV heads — GQA group ratio = 4)
head_dim:               64
num_hidden_layers:      16
intermediate_size:      8192
vocab_size:             128256
rope_theta:             500000.0
rms_norm_eps:           1e-05
tie_word_embeddings:    true
BOS token:              128000
EOS tokens:             128001, 128008, 128009
```

#### Why inspect and assert shapes before writing any model code?

Wrong tensor shapes or names discovered at inference time produce confusing matmul errors. Discovered at weight-load time (with a name→shape assertion table) they pinpoint the exact misconfiguration. The assert script exits 0 only when every expected tensor exists with the correct shape — this is a one-time investment that saves hours of debugging in Phase 1.

#### Why torch backend for safetensors, not numpy?

Llama 3.2 weights are stored as bfloat16 (Brain Float 16), a 16-bit float format developed by Google Brain. NumPy has no bfloat16 dtype. PyTorch supports it natively. For Phase 1 reference math (NumPy forward pass), weights are cast to float32 immediately after loading — bfloat16 is only the storage format, not the compute format for the reference implementation.

---

## Phase 0.3 — Tokenizer Round-Trip

*Not yet completed.*

---

## Phase 0.4 — PACE Environment Bootstrap

*Not yet completed — parallel track on PACE Phoenix.*

---

## Phase 0.5 — nanobind Round-Trip

*Not yet completed — depends on 0.4.*

---

## Phase 1 — Correctness

*Not yet started.*

---

## Phase 2 — Usable Engine

*Not yet started.*

---

## Phase 3 — Baseline Benchmarks

*Not yet started.*

---

## Phase 4 — Differentiators

*Not yet started.*

---

## Phase 5 — Polish

*Not yet started.*

---

## Reference: The From-Scratch Boundary

Per PRD §2, this is what counts as "from-scratch" vs "library is fine":

| Component | From-scratch (you implement) | Library (use freely) |
|-----------|-----------------------------|--------------------|
| Embedding lookup | ✅ | |
| RMSNorm | ✅ | |
| RoPE | ✅ | |
| GQA attention | ✅ | |
| SwiGLU FFN | ✅ | |
| KV cache | ✅ | |
| Scheduler / batching | ✅ | |
| Paged memory manager | ✅ | |
| Quantization logic | ✅ | |
| CUDA attention kernel | ✅ | |
| Array storage (NumPy/torch tensors) | | ✅ |
| GEMM (cuBLAS) | | ✅ |
| Tokenizer | | ✅ (HF tokenizers) |
| Weight parsing | | ✅ (safetensors) |

This boundary is what makes the project interview-credible: you are not reimplementing BLAS, you are implementing the model logic and the serving system.

---

## Reference: Llama 3.2 1B Architecture

Source: `weights/config.json`. Full details in `notes/model_config.md`.

- `hidden_size` — 2048
- `num_attention_heads` — 32
- `num_key_value_heads` — 8
- `num_hidden_layers` — 16
- `intermediate_size` — 8192
- `rope_theta` — 500000.0 (plus Llama3 scaling — see notes)
- `rms_norm_eps` — 1e-05
- `vocab_size` — 128256
- `tie_word_embeddings` — true (no separate lm_head tensor)
- GQA group ratio — 4 (32 query heads / 8 KV heads)
