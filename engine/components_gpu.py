"""
Torch GPU implementations of all Llama components (Phase 3.2).

Mirrors engine/components.py but uses fp16 torch tensors on cuda:0.
RoPE tables are fp32 for precision; all activations are fp16.
"""

import torch
import torch.nn.functional as F


def rms_norm_gpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 internally for numerical safety, returns fp16."""
    x32 = x.float()
    rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x32 / rms) * weight.float()).half()


def _llama3_inv_freq_gpu(head_dim: int, theta: float, rope_scaling: dict) -> torch.Tensor:
    """Llama3 frequency scaling — runs on CPU, returns float32 tensor."""
    factor           = rope_scaling["factor"]
    low_freq_factor  = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    orig_max         = rope_scaling["original_max_position_embeddings"]

    i        = torch.arange(0, head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (theta ** (i / head_dim))

    low_wavelen  = orig_max / low_freq_factor
    high_wavelen = orig_max / high_freq_factor

    new_inv_freq = torch.empty_like(inv_freq)
    for j in range(len(inv_freq)):
        freq    = inv_freq[j].item()
        wavelen = 2.0 * 3.141592653589793 / freq
        if wavelen < high_wavelen:
            new_inv_freq[j] = freq
        elif wavelen > low_wavelen:
            new_inv_freq[j] = freq / factor
        else:
            smooth = (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freq[j] = (1.0 - smooth) * (freq / factor) + smooth * freq

    return new_inv_freq.float()


def precompute_rope_tables_gpu(
    max_seq: int,
    head_dim: int,
    theta: float,
    rope_scaling=None,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin RoPE tables as fp32 tensors on device.
    Returns: cos, sin — each shape (max_seq, head_dim)
    """
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        inv_freq = _llama3_inv_freq_gpu(head_dim, theta, rope_scaling)
    else:
        i        = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (i / head_dim))

    positions = torch.arange(max_seq, dtype=torch.float32)
    angles    = torch.outer(positions, inv_freq)                        # (max_seq, head_dim//2)
    cos       = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(device)
    sin       = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(device)
    return cos, sin


def apply_rope_gpu(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x — shape (seq, n_heads, head_dim).
    cos/sin are fp32 (seq, head_dim); x is fp16.
    Computes in fp32, returns fp16.
    """
    d     = x.shape[-1]
    x32   = x.float()
    x_rot = torch.cat([-x32[..., d // 2:], x32[..., :d // 2]], dim=-1)
    cos_  = cos[:, None, :]   # (seq, 1, head_dim) — broadcast over heads
    sin_  = sin[:, None, :]
    return (x32 * cos_ + x_rot * sin_).half()


def gqa_attention_gpu(
    x: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    o_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    kv_cache=None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    Grouped-Query Attention — GPU fp16 version.
    Mirrors gqa_attention() from components.py exactly.
    """
    seq    = x.shape[0]
    groups = n_heads // n_kv_heads

    q = (x @ q_w.T).reshape(seq, n_heads,    head_dim)
    k = (x @ k_w.T).reshape(seq, n_kv_heads, head_dim)
    v = (x @ v_w.T).reshape(seq, n_kv_heads, head_dim)

    cos_pos = cos[positions]   # (seq, head_dim)
    sin_pos = sin[positions]
    q = apply_rope_gpu(q, cos_pos, sin_pos)
    k = apply_rope_gpu(k, cos_pos, sin_pos)

    if kv_cache is not None:
        read_len = kv_cache.pos + seq
        kv_cache.k[layer_idx, kv_cache.pos:read_len] = k
        kv_cache.v[layer_idx, kv_cache.pos:read_len] = v
        k_full = kv_cache.k[layer_idx, :read_len]
        v_full = kv_cache.v[layer_idx, :read_len]
    else:
        k_full = k
        v_full = v

    kv_seq = k_full.shape[0]

    k_full = torch.repeat_interleave(k_full, groups, dim=1)   # (kv_seq, n_heads, head_dim)
    v_full = torch.repeat_interleave(v_full, groups, dim=1)

    q      = q.transpose(0, 1)        # (n_heads, seq, head_dim)
    k_full = k_full.transpose(0, 1)   # (n_heads, kv_seq, head_dim)
    v_full = v_full.transpose(0, 1)

    scale  = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k_full.float().transpose(1, 2)) * scale  # fp32 for stability

    if seq > 1:
        offset = kv_seq - seq
        mask   = torch.triu(torch.full((seq, kv_seq), float("-inf"), device=x.device), diagonal=offset + 1)
        scores = scores + mask.unsqueeze(0)

    scores = F.softmax(scores, dim=-1).half()

    out = torch.matmul(scores, v_full)   # (n_heads, seq, head_dim)
    out = out.transpose(0, 1)            # (seq, n_heads, head_dim)
    out = out.reshape(seq, n_heads * head_dim)

    return out @ o_w.T


def swiglu_ffn_gpu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU FFN: down( silu(gate(x)) * up(x) )"""
    gate = F.silu(x @ gate_w.T)
    up   = x @ up_w.T
    return (gate * up) @ down_w.T
