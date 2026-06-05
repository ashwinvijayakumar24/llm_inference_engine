"""
Quantization tests.

Fast tests — pure torch, run on CPU (Mac) or GPU. No real weights. Validate the
quant MATH and the linear() chokepoint.

Slow tests (@slow) — real 2.4GB weights on CUDA (PACE). Validate that a quantized
model produces sane output. Run with: pytest tests/test_quant.py -v -m slow
"""

import pytest
import torch

from engine.quant import (
    QuantWeight,
    quantize_int8_perchannel,
    dequantize_int8_perchannel,
    quantize_int4_group,
    dequantize_int4_group,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
slow = pytest.mark.slow

WEIGHTS_PATH = "weights"
TOKEN_IDS    = [128000, 9906, 358, 1097]   # <bos> Hello I am
VOCAB_SIZE   = 128256


# ---------------------------------------------------------------------------
# int8 per-channel
# ---------------------------------------------------------------------------

def test_int8_roundtrip_error_bounded():
    """Dequantized weight is within one quantization step of the original."""
    torch.manual_seed(0)
    w = torch.randn(64, 32)

    q, scale = quantize_int8_perchannel(w)
    w_hat    = dequantize_int8_perchannel(q, scale).float()

    # Per element: rounding error <= scale/2, PLUS fp16-scale storage error.
    # scale is stored fp16; q*scale_fp16 differs from q*scale_fp32 by up to
    # |q| * scale * 2^-11 <= 127 * scale * 2^-11 ~= 0.06*scale. Bound 0.75*scale is safe.
    per_row_bound = scale.float() * 0.75 + 1e-3
    err = (w - w_hat).abs()
    assert torch.all(err <= per_row_bound[:, None]), \
        f"max err {err.max():.4f} exceeds bound {per_row_bound.max():.4f}"


def test_int8_shapes_and_dtype():
    w = torch.randn(10, 8)
    q, scale = quantize_int8_perchannel(w)
    assert q.shape == (10, 8)
    assert q.dtype == torch.int8
    assert scale.shape == (10,)
    assert q.abs().max() <= 127


def test_int8_zero_row_no_nan():
    """A row of all zeros must not produce NaN (scale guard)."""
    w = torch.randn(4, 16)
    w[2] = 0.0
    q, scale = quantize_int8_perchannel(w)
    w_hat = dequantize_int8_perchannel(q, scale)
    assert torch.all(torch.isfinite(w_hat))
    assert torch.all(w_hat[2] == 0.0)


# ---------------------------------------------------------------------------
# int4 group-wise
# ---------------------------------------------------------------------------

def test_int4_pack_unpack_identity():
    """Packing then unpacking recovers the exact int4 values (before scaling)."""
    torch.manual_seed(1)
    # Build a weight whose quantized values we can recompute directly.
    w = torch.randn(8, 128)
    group_size = 32

    packed, scale = quantize_int4_group(w, group_size=group_size)
    assert packed.dtype == torch.uint8
    assert packed.shape == (8, 64)              # in // 2
    assert scale.shape == (8, 128 // group_size)

    # Recompute expected int4 quant values (as int — int has no signed zero).
    w32 = w.float().reshape(8, 128 // group_size, group_size)
    q_expected = torch.round(w32 / scale.float()[:, :, None]).clamp(-7, 7).to(torch.int16)
    q_expected = q_expected.reshape(8, 128)

    # Unpack manually and sign-extend.
    low  = (packed & 0xF).to(torch.int16)
    high = ((packed >> 4) & 0xF).to(torch.int16)
    q = torch.empty(8, 128, dtype=torch.int16)
    q[:, 0::2] = low
    q[:, 1::2] = high
    q = torch.where(q >= 8, q - 16, q)

    assert torch.equal(q, q_expected), "pack/unpack changed int4 values"


def test_int4_roundtrip_error_bounded():
    torch.manual_seed(2)
    w = torch.randn(16, 256)
    group_size = 128

    packed, scale = quantize_int4_group(w, group_size=group_size)
    w_hat = dequantize_int4_group(packed, scale, group_size).float()

    # int4 step is larger than int8 — bound is scale/2 per group.
    scale_exp = scale.float().repeat_interleave(group_size, dim=1)  # (out, in)
    bound = scale_exp * 0.5 + 1e-3
    err = (w - w_hat).abs()
    assert torch.all(err <= bound), f"max err {err.max():.4f} exceeds bound {bound.max():.4f}"


def test_int4_memory_smaller_than_int8():
    """int4 stored bytes < int8 stored bytes < fp16 bytes."""
    w = torch.randn(128, 512)

    q8, s8   = quantize_int8_perchannel(w)
    qw8      = QuantWeight(q8, s8, "int8")

    p4, s4   = quantize_int4_group(w, group_size=128)
    qw4      = QuantWeight(p4, s4, "int4", 128)

    fp16_bytes = w.half().element_size() * w.nelement()
    assert qw8.nbytes() < fp16_bytes
    assert qw4.nbytes() < qw8.nbytes()


# ---------------------------------------------------------------------------
# QuantWeight container
# ---------------------------------------------------------------------------

def test_quantweight_dequantize_int8():
    w = torch.randn(8, 16)
    q, scale = quantize_int8_perchannel(w)
    qw = QuantWeight(q, scale, "int8")
    assert qw.dequantize().shape == (8, 16)
    assert qw.dequantize().dtype == torch.float16


def test_quantweight_dequantize_int4():
    w = torch.randn(8, 128)
    p, scale = quantize_int4_group(w, group_size=64)
    qw = QuantWeight(p, scale, "int4", 64)
    assert qw.dequantize().shape == (8, 128)
    assert qw.dequantize().dtype == torch.float16


# ---------------------------------------------------------------------------
# linear() chokepoint
# ---------------------------------------------------------------------------

def test_linear_plain_weight_bit_identical():
    """linear(x, w) with a plain tensor must equal x @ w.T exactly (fp16 path untouched)."""
    from engine.components_gpu import linear
    torch.manual_seed(3)
    x = torch.randn(4, 16, dtype=torch.float16)
    w = torch.randn(8, 16, dtype=torch.float16)
    assert torch.equal(linear(x, w), x @ w.T)


def test_linear_quantweight_matches_dequant():
    """linear(x, QuantWeight) equals x @ dequant(W).T."""
    from engine.components_gpu import linear
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float16)
    w = torch.randn(8, 32)
    q, scale = quantize_int8_perchannel(w)
    qw = QuantWeight(q, scale, "int8")
    assert torch.equal(linear(x, qw), x @ qw.dequantize().T)


# ---------------------------------------------------------------------------
# Slow — real weights on CUDA (PACE)
# ---------------------------------------------------------------------------

@cuda_only
@slow
def test_int8_model_argmax_matches_fp16():
    """
    int8-quantized model's first-token argmax must match the fp16 model.
    A wrong scale axis produces fluent-but-wrong text — this catches it.
    Loads one model at a time to avoid OOM.
    """
    from engine.loader import load_config, load_weights_gpu, load_weights_gpu_quant
    from engine.model_gpu import LlamaModelGPU
    import numpy as np

    config = load_config(WEIGHTS_PATH)

    w_fp16   = load_weights_gpu(WEIGHTS_PATH, config)
    model    = LlamaModelGPU(w_fp16, config)
    logits_f = model.prefill(TOKEN_IDS, model.make_cache(2048))
    fp16_arg = int(np.argmax(logits_f))
    del w_fp16, model
    torch.cuda.empty_cache()

    w_int8   = load_weights_gpu_quant(WEIGHTS_PATH, config, mode="int8")
    model    = LlamaModelGPU(w_int8, config)
    logits_q = model.prefill(TOKEN_IDS, model.make_cache(2048))

    assert logits_q.shape == (VOCAB_SIZE,)
    assert np.all(np.isfinite(logits_q))
    assert int(np.argmax(logits_q)) == fp16_arg, (
        f"int8 argmax {int(np.argmax(logits_q))} != fp16 argmax {fp16_arg} "
        "(likely wrong quant scale axis)"
    )
