"""
Weight-only quantization (Phase 4.1).

Two schemes, both symmetric (no zero-point):
- int8 per-channel: one scale per output row of the weight matrix
- int4 group-wise:  one scale per group of `group_size` input columns, two int4 packed per byte

Weights are stored (out, in) and applied as `x @ W.T`. "Per-channel" = per output
row (axis 0) because each output channel is an independent dot product over the
input — giving each its own scale minimizes error where it matters.

All functions operate on the input tensor's existing device/dtype — no hardcoded
cuda, so tests run on CPU (Mac) and production runs on cuda:0.

Dequantize-on-the-fly: store q + scale, reconstruct the fp16 weight at matmul time.
This isolates the quantization MATH from kernel performance (the CUDA kernel handles
speed separately).
"""

import torch

_EPS = 1e-8  # guards against a zero scale when an entire row/group is zero


# ---------------------------------------------------------------------------
# int8 per-channel (per output row)
# ---------------------------------------------------------------------------

def quantize_int8_perchannel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w: (out, in) float tensor.
    Returns (q_int8, scale) where:
        q_int8: (out, in)  int8 in [-127, 127]
        scale:  (out,)     fp16 — scale[r] reconstructs row r
    """
    w32   = w.float()
    scale = w32.abs().amax(dim=1) / 127.0          # (out,)
    scale = scale.clamp(min=_EPS)
    # Round using the fp16-rounded scale (the same value dequant will use) so
    # quant/dequant are consistent and there is no fp32-vs-fp16 tie ambiguity.
    scale = scale.half().float()
    q     = torch.round(w32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.half()


def dequantize_int8_perchannel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8_perchannel. Returns (out, in) fp16."""
    return (q.float() * scale.float()[:, None]).half()


# ---------------------------------------------------------------------------
# int4 group-wise (per group of input columns)
# ---------------------------------------------------------------------------

def quantize_int4_group(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w: (out, in) float tensor; `in` must be divisible by group_size.
    Symmetric int4 in [-7, 7], two values packed per uint8 byte.

    Returns (q_packed, scale):
        q_packed: (out, in // 2)            uint8 — low nibble first, then high
        scale:    (out, in // group_size)   fp16 — one scale per group
    """
    out, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    assert in_dim % 2 == 0, "in_dim must be even to pack two int4 per byte"

    w32    = w.float().reshape(out, in_dim // group_size, group_size)
    scale  = w32.abs().amax(dim=2) / 7.0                       # (out, n_groups)
    scale  = scale.clamp(min=_EPS)
    # Round with the fp16-rounded scale dequant will use (avoids tie ambiguity).
    scale  = scale.half().float()
    q      = torch.round(w32 / scale[:, :, None]).clamp(-7, 7).to(torch.int8)
    q      = q.reshape(out, in_dim)                            # (out, in)

    # Pack two signed int4 per byte: low nibble = even index, high nibble = odd.
    # Store as unsigned nibbles (& 0xF); sign is recovered on unpack.
    nibbles = (q & 0xF).to(torch.uint8)                        # (out, in)
    low     = nibbles[:, 0::2]                                 # (out, in//2)
    high    = nibbles[:, 1::2]
    packed  = (low | (high << 4)).to(torch.uint8)
    return packed, scale.half()


def dequantize_int4_group(q_packed: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Inverse of quantize_int4_group. Returns (out, in) fp16."""
    out, half_in = q_packed.shape
    in_dim       = half_in * 2

    low  = (q_packed & 0xF).to(torch.int16)        # (out, in//2)
    high = ((q_packed >> 4) & 0xF).to(torch.int16)

    # Interleave back to (out, in): even indices from low, odd from high.
    q = torch.empty((out, in_dim), dtype=torch.int16, device=q_packed.device)
    q[:, 0::2] = low
    q[:, 1::2] = high

    # Sign-extend 4-bit: values >= 8 represent negatives (v - 16).
    q = torch.where(q >= 8, q - 16, q).float()

    q = q.reshape(out, in_dim // group_size, group_size)
    w = q * scale.float()[:, :, None]
    return w.reshape(out, in_dim).half()


# ---------------------------------------------------------------------------
# QuantWeight container
# ---------------------------------------------------------------------------

class QuantWeight:
    """
    Holds a quantized linear weight. The forward path checks `isinstance(w, QuantWeight)`
    and dequantizes on the fly. `mode` is "int8" or "int4".
    """

    __slots__ = ("q", "scale", "mode", "group_size")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, mode: str, group_size: int = 128):
        self.q          = q
        self.scale      = scale
        self.mode       = mode
        self.group_size = group_size

    def dequantize(self) -> torch.Tensor:
        if self.mode == "int8":
            return dequantize_int8_perchannel(self.q, self.scale)
        elif self.mode == "int4":
            return dequantize_int4_group(self.q, self.scale, self.group_size)
        raise ValueError(f"Unknown quant mode: {self.mode}")

    def nbytes(self) -> int:
        """Stored size in bytes (q + scale) — for the memory-delta benchmark."""
        return self.q.element_size() * self.q.nelement() + \
               self.scale.element_size() * self.scale.nelement()
