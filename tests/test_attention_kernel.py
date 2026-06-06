"""
CUDA decode-attention kernel tests.

HARD GATE (task 4.4.3): 100 random inputs, varying kv_seq, max-abs-diff < 1e-3
vs the torch reference. Do not optimize past a stage until this is green.

Runs on PACE (needs CUDA + the built engine_kernels module). Skips cleanly
elsewhere. Build first: bash scripts/build_kernels.sh
"""

import sys
from pathlib import Path

import pytest
import torch

# Make the compiled module (build/) and the kernels/ reference importable.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "build"))
sys.path.insert(0, str(_ROOT / "kernels"))

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

try:
    import engine_kernels  # noqa: F401
    _HAVE_KERNELS = True
except ImportError:
    _HAVE_KERNELS = False

needs_kernels = pytest.mark.skipif(not _HAVE_KERNELS, reason="engine_kernels not built")

# Llama 3.2 1B decode shapes.
N_HEADS  = 32
N_KV     = 8
HEAD_DIM = 64
SCALE    = 1.0 / (HEAD_DIM ** 0.5)
DEV      = "cuda:0"


def _rand_inputs(kv_seq, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Small scale keeps fp16 well-conditioned (same trick as component tests).
    q = (torch.randn(N_HEADS, HEAD_DIM, generator=g) * 0.1).half().to(DEV)
    k = (torch.randn(N_KV, kv_seq, HEAD_DIM, generator=g) * 0.1).half().to(DEV)
    v = (torch.randn(N_KV, kv_seq, HEAD_DIM, generator=g) * 0.1).half().to(DEV)
    return q, k, v


VERSIONS = ["v1", "v2"]


@cuda_only
@needs_kernels
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kv_seq", [1, 7, 64, 333, 2048])
def test_matches_reference_across_seqlens(version, kv_seq):
    """Diff kernel vs torch reference at several kv_seq lengths."""
    from attn_reference import attention_decode_reference, attention_decode

    q, k, v = _rand_inputs(kv_seq, seed=kv_seq)
    ref = attention_decode_reference(q, k, v, SCALE)
    got = attention_decode(q, k, v, SCALE, version=version)

    diff = (got.float() - ref.float()).abs().max().item()
    assert diff < 1e-3, f"{version} kv_seq={kv_seq}: max-abs-diff {diff:.2e} >= 1e-3"


@cuda_only
@needs_kernels
@pytest.mark.parametrize("version", VERSIONS)
def test_hard_gate_100_random_inputs(version):
    """Task 4.4.3 hard gate: 100 random inputs, every one must pass < 1e-3."""
    from attn_reference import attention_decode_reference, attention_decode

    worst = 0.0
    for seed in range(100):
        kv_seq = 1 + (seed * 37) % 512          # spread 1..512
        q, k, v = _rand_inputs(kv_seq, seed=1000 + seed)
        ref = attention_decode_reference(q, k, v, SCALE)
        got = attention_decode(q, k, v, SCALE, version=version)
        diff = (got.float() - ref.float()).abs().max().item()
        worst = max(worst, diff)
        assert diff < 1e-3, f"{version} seed={seed} kv_seq={kv_seq}: diff {diff:.2e} >= 1e-3"
    print(f"\n  {version} worst diff across 100 inputs: {worst:.2e}")


@cuda_only
@needs_kernels
@pytest.mark.parametrize("version", VERSIONS)
def test_gqa_mapping(version):
    """
    Query head h must read KV head h//groups. Build K/V where each KV head has a
    distinct constant value; the output per query head reveals which KV head it used.
    """
    from attn_reference import attention_decode

    groups = N_HEADS // N_KV   # 4
    kv_seq = 4

    q = torch.zeros(N_HEADS, HEAD_DIM, dtype=torch.float16, device=DEV)
    # V[kv_head] is constant = kv_head + 1 everywhere; with any softmax weights,
    # the output for a query head equals its KV head's constant.
    v = torch.empty(N_KV, kv_seq, HEAD_DIM, dtype=torch.float16, device=DEV)
    for kvh in range(N_KV):
        v[kvh] = float(kvh + 1)
    k = torch.zeros(N_KV, kv_seq, HEAD_DIM, dtype=torch.float16, device=DEV)

    out = attention_decode(q, k, v, SCALE, version=version).float()
    for h in range(N_HEADS):
        expected = float(h // groups + 1)
        assert torch.allclose(out[h], torch.full_like(out[h], expected), atol=1e-2), \
            f"{version} head {h} read wrong KV head (got {out[h,0].item()}, expected {expected})"
