"""
Fast GPU component tests — no real weights, all synthetic random tensors.
All tests skip automatically if CUDA is not available.
atol=1e-2 throughout: fp16 GPU vs fp32 CPU difference can reach ~0.01.
"""

import numpy as np
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEV = "cuda:0"


# ---------------------------------------------------------------------------
# Imports (deferred so CPU-only machines can still collect test names)
# ---------------------------------------------------------------------------

def _cpu():
    from engine.components import (
        rms_norm, precompute_rope_tables, apply_rope,
        swiglu_ffn, gqa_attention,
    )
    return rms_norm, precompute_rope_tables, apply_rope, swiglu_ffn, gqa_attention


def _gpu():
    from engine.components_gpu import (
        rms_norm_gpu, precompute_rope_tables_gpu, apply_rope_gpu,
        swiglu_ffn_gpu, gqa_attention_gpu,
    )
    return rms_norm_gpu, precompute_rope_tables_gpu, apply_rope_gpu, swiglu_ffn_gpu, gqa_attention_gpu


# ---------------------------------------------------------------------------
# KVCacheGPU
# ---------------------------------------------------------------------------

@cuda_only
def test_kv_cache_gpu_shapes():
    from engine.cache import KVCacheGPU
    cache = KVCacheGPU(n_layers=4, max_seq=16, n_kv_heads=2, head_dim=8, device=DEV)
    assert cache.k.shape == (4, 16, 2, 8)
    assert cache.v.shape == (4, 16, 2, 8)
    assert cache.k.dtype == torch.float16
    assert cache.k.device.type == "cuda"
    assert cache.pos == 0


@cuda_only
def test_kv_cache_gpu_advance():
    from engine.cache import KVCacheGPU
    cache = KVCacheGPU(n_layers=2, max_seq=32, n_kv_heads=2, head_dim=8, device=DEV)
    cache.advance(5)
    assert cache.pos == 5
    cache.advance(1)
    assert cache.pos == 6


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@cuda_only
def test_rms_norm_gpu_vs_cpu():
    rms_norm, *_ = _cpu()
    rms_norm_gpu, *_ = _gpu()

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8, 64)).astype(np.float32)
    w_np = rng.standard_normal((64,)).astype(np.float32)
    eps  = 1e-5

    cpu_out = rms_norm(x_np, w_np, eps)

    x_t = torch.tensor(x_np, dtype=torch.float16, device=DEV)
    w_t = torch.tensor(w_np, dtype=torch.float16, device=DEV)
    gpu_out = rms_norm_gpu(x_t, w_t, eps).cpu().float().numpy()

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-2)


# ---------------------------------------------------------------------------
# RoPE tables
# ---------------------------------------------------------------------------

@cuda_only
def test_precompute_rope_tables_gpu_matches_cpu():
    _, precompute_rope_tables, *_ = _cpu()
    _, precompute_rope_tables_gpu, *_ = _gpu()

    cos_cpu, sin_cpu = precompute_rope_tables(max_seq=64, head_dim=32, theta=500000.0)
    cos_gpu, sin_gpu = precompute_rope_tables_gpu(max_seq=64, head_dim=32, theta=500000.0, device=DEV)

    np.testing.assert_allclose(cos_gpu.cpu().numpy(), cos_cpu, atol=1e-4)
    np.testing.assert_allclose(sin_gpu.cpu().numpy(), sin_cpu, atol=1e-4)


# ---------------------------------------------------------------------------
# apply_rope
# ---------------------------------------------------------------------------

@cuda_only
def test_apply_rope_gpu_vs_cpu():
    _, precompute_rope_tables, apply_rope, *_ = _cpu()
    _, precompute_rope_tables_gpu, apply_rope_gpu, *_ = _gpu()

    rng  = np.random.default_rng(1)
    x_np = rng.standard_normal((4, 8, 32)).astype(np.float32)
    cos_cpu, sin_cpu = precompute_rope_tables(max_seq=4, head_dim=32, theta=500000.0)

    cpu_out = apply_rope(x_np, cos_cpu, sin_cpu)

    x_t     = torch.tensor(x_np, dtype=torch.float16, device=DEV)
    cos_gpu, sin_gpu = precompute_rope_tables_gpu(max_seq=4, head_dim=32, theta=500000.0, device=DEV)
    gpu_out = apply_rope_gpu(x_t, cos_gpu, sin_gpu).cpu().float().numpy()

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-2)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

@cuda_only
def test_swiglu_ffn_gpu_vs_cpu():
    *_, swiglu_ffn, _ = _cpu()
    *_, swiglu_ffn_gpu, _ = _gpu()

    rng    = np.random.default_rng(2)
    seq, H, FF = 4, 16, 32
    x_np      = rng.standard_normal((seq, H)).astype(np.float32)
    gate_np   = rng.standard_normal((FF, H)).astype(np.float32)
    up_np     = rng.standard_normal((FF, H)).astype(np.float32)
    down_np   = rng.standard_normal((H,  FF)).astype(np.float32)

    cpu_out = swiglu_ffn(x_np, gate_np, up_np, down_np)

    def t(a): return torch.tensor(a, dtype=torch.float16, device=DEV)
    gpu_out = swiglu_ffn_gpu(t(x_np), t(gate_np), t(up_np), t(down_np)).cpu().float().numpy()

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-2)


# ---------------------------------------------------------------------------
# GQA attention (no cache, small toy dims)
# ---------------------------------------------------------------------------

@cuda_only
def test_gqa_attention_gpu_vs_cpu():
    *_, gqa_attention = _cpu()
    *_, gqa_attention_gpu = _gpu()
    _, precompute_rope_tables, *_ = _cpu()
    _, precompute_rope_tables_gpu, *_ = _gpu()

    rng = np.random.default_rng(3)
    seq, H, NH, NKV, HD = 6, 32, 4, 2, 8
    scale = 0.02  # small scale to keep fp16 in range

    x_np  = (rng.standard_normal((seq, H)) * scale).astype(np.float32)
    q_np  = (rng.standard_normal((NH * HD, H)) * scale).astype(np.float32)
    k_np  = (rng.standard_normal((NKV * HD, H)) * scale).astype(np.float32)
    v_np  = (rng.standard_normal((NKV * HD, H)) * scale).astype(np.float32)
    o_np  = (rng.standard_normal((H, NH * HD)) * scale).astype(np.float32)

    cos_cpu, sin_cpu = precompute_rope_tables(max_seq=seq, head_dim=HD, theta=10000.0)
    positions_np = np.arange(seq, dtype=np.int32)

    cpu_out = gqa_attention(x_np, q_np, k_np, v_np, o_np,
                            cos_cpu, sin_cpu, positions_np, NH, NKV, HD)

    def t(a): return torch.tensor(a, dtype=torch.float16, device=DEV)
    cos_gpu, sin_gpu = precompute_rope_tables_gpu(max_seq=seq, head_dim=HD, theta=10000.0, device=DEV)
    positions_t = torch.arange(seq, dtype=torch.long, device=DEV)

    gpu_out = gqa_attention_gpu(
        t(x_np), t(q_np), t(k_np), t(v_np), t(o_np),
        cos_gpu, sin_gpu, positions_t, NH, NKV, HD
    ).cpu().float().numpy()

    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-2)
