"""
From-scratch NumPy implementations of every Llama 3.2 component.

Every function here is a pure function: same inputs → same outputs, no side effects.
Each one is diffed against HuggingFace in tests/test_components.py.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    Root Mean Square Layer Normalization.
    Formula: x * weight / sqrt(mean(x^2) + eps)
    eps is INSIDE the sqrt — putting it outside is the classic bug.
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding) with Llama 3 scaling
# ---------------------------------------------------------------------------

def _llama3_inv_freq(head_dim: int, theta: float, rope_scaling: dict) -> np.ndarray:
    """
    Compute inverse frequencies with Llama 3 frequency-dependent scaling.

    Low-frequency components (long wavelengths) are scaled down by `factor`
    to extend context. High-frequency components are unchanged. Smooth blend
    between. Matches HF LlamaRotaryEmbedding with rope_type='llama3'.
    """
    factor          = rope_scaling["factor"]                           # 32.0
    low_freq_factor = rope_scaling["low_freq_factor"]                  # 1.0
    high_freq_factor= rope_scaling["high_freq_factor"]                 # 4.0
    orig_max        = rope_scaling["original_max_position_embeddings"] # 8192

    # Standard base frequencies: theta^(-2i/d) for i = 0, 2, 4, ...
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (i / head_dim))

    low_wavelen  = orig_max / low_freq_factor   # 8192 — below this: scale
    high_wavelen = orig_max / high_freq_factor  # 2048 — above this: no scale

    new_inv_freq = np.empty_like(inv_freq)
    for j, freq in enumerate(inv_freq):
        wavelen = 2.0 * np.pi / freq
        if wavelen < high_wavelen:
            # High frequency — unchanged
            new_inv_freq[j] = freq
        elif wavelen > low_wavelen:
            # Low frequency — scale down by factor (extends context)
            new_inv_freq[j] = freq / factor
        else:
            # Smooth blend between scaled and unscaled
            smooth = (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freq[j] = (1.0 - smooth) * (freq / factor) + smooth * freq

    return new_inv_freq.astype(np.float32)


def precompute_rope_tables(
    max_seq: int,
    head_dim: int,
    theta: float,
    rope_scaling: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos/sin tables for RoPE.

    Returns:
        cos: shape (max_seq, head_dim)
        sin: shape (max_seq, head_dim)

    Tables are indexed by position: cos[pos] gives the rotation for that position.
    head_dim entries cover both halves (repeated) for the half-rotation layout.
    """
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        inv_freq = _llama3_inv_freq(head_dim, theta, rope_scaling)
    else:
        i = np.arange(0, head_dim, 2, dtype=np.float32)
        inv_freq = (1.0 / (theta ** (i / head_dim))).astype(np.float32)

    # positions × inv_freq → angles of shape (max_seq, head_dim // 2)
    positions = np.arange(max_seq, dtype=np.float32)
    angles = np.outer(positions, inv_freq)  # (max_seq, head_dim//2)

    # Duplicate for both halves: shape (max_seq, head_dim)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)

    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embedding to query or key tensor.

    Args:
        x:   shape (seq, n_heads, head_dim)
        cos: shape (seq, head_dim)  — already sliced to correct positions
        sin: shape (seq, head_dim)

    Llama uses half-rotation layout (NOT interleaved):
        x_rot = [-x[..., d/2:], x[..., :d/2]]
    so the rotation acts on each head as:
        x_out = x * cos + x_rot * sin
    """
    d = x.shape[-1]
    x_rot = np.concatenate([-x[..., d // 2:], x[..., :d // 2]], axis=-1)
    # Broadcast cos/sin from (seq, head_dim) → (seq, 1, head_dim)
    cos = cos[:, np.newaxis, :]
    sin = sin[:, np.newaxis, :]
    return x * cos + x_rot * sin


# ---------------------------------------------------------------------------
# GQA Attention
# ---------------------------------------------------------------------------

def gqa_attention(
    x: np.ndarray,
    q_w: np.ndarray,
    k_w: np.ndarray,
    v_w: np.ndarray,
    o_w: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    positions: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    kv_cache=None,
    layer_idx: int | None = None,
) -> np.ndarray:
    """
    Grouped-Query Attention (GQA) forward pass.

    Args:
        x:          (seq, hidden)
        q_w:        (n_heads * head_dim, hidden)
        k_w:        (n_kv_heads * head_dim, hidden)
        v_w:        (n_kv_heads * head_dim, hidden)
        o_w:        (hidden, n_heads * head_dim)
        cos, sin:   (max_seq, head_dim) — full tables; indexed at positions
        positions:  (seq,) int array — position indices for RoPE
        kv_cache:   KVCache instance (Phase 2+); None uses no-cache path
        layer_idx:  layer index into kv_cache (required when kv_cache is not None)

    Returns:
        (seq, hidden)

    KV cache path (Phase 2):
        - Writes new K/V into cache at cache.pos..cache.pos+seq
        - Reads full K/V from 0..cache.pos+seq for attention
        - advance() must be called by the caller after all layers complete
    """
    seq    = x.shape[0]
    groups = n_heads // n_kv_heads

    # Project and reshape
    q = (x @ q_w.T).reshape(seq, n_heads,    head_dim)   # (seq, NH, HD)
    k = (x @ k_w.T).reshape(seq, n_kv_heads, head_dim)   # (seq, NKV, HD)
    v = (x @ v_w.T).reshape(seq, n_kv_heads, head_dim)   # (seq, NKV, HD)

    # Apply RoPE — slice tables at the given positions
    cos_pos = cos[positions]   # (seq, head_dim)
    sin_pos = sin[positions]
    q = apply_rope(q, cos_pos, sin_pos)
    k = apply_rope(k, cos_pos, sin_pos)

    if kv_cache is not None:
        # Write new K/V to cache, then read full history for attention
        read_len = kv_cache.pos + seq
        kv_cache.k[layer_idx, kv_cache.pos:read_len] = k
        kv_cache.v[layer_idx, kv_cache.pos:read_len] = v
        k_full = kv_cache.k[layer_idx, :read_len]   # (read_len, NKV, HD)
        v_full = kv_cache.v[layer_idx, :read_len]
    else:
        k_full = k
        v_full = v

    kv_seq = k_full.shape[0]

    # GQA broadcast: each KV head serves `groups` query heads
    k_full = np.repeat(k_full, groups, axis=1)   # (kv_seq, NH, HD)
    v_full = np.repeat(v_full, groups, axis=1)

    # Batched attention: transpose to (NH, seq/kv_seq, HD) for matmul
    q      = q.transpose(1, 0, 2)        # (NH, seq, HD)
    k_full = k_full.transpose(1, 0, 2)   # (NH, kv_seq, HD)
    v_full = v_full.transpose(1, 0, 2)

    # Scaled dot-product scores: (NH, seq, kv_seq)
    scale  = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k_full.transpose(0, 2, 1)) * scale

    # Causal mask: only needed for multi-token forward (prefill).
    # For decode (seq==1) all kv positions are already in the past — no mask.
    # np.triu with k=offset+1 makes query qi attend to kv columns 0..offset+qi.
    if seq > 1:
        offset = kv_seq - seq  # prior tokens in cache (0 for no-cache path)
        mask   = np.triu(np.full((seq, kv_seq), float("-inf"), dtype=np.float32), k=offset + 1)
        scores = scores + mask[np.newaxis]

    # Numerically stable softmax along key axis
    scores = scores - scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores)
    scores = scores / scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    out = np.matmul(scores, v_full)   # (NH, seq, HD)
    out = out.transpose(1, 0, 2)      # (seq, NH, HD)
    out = out.reshape(seq, n_heads * head_dim)

    return out @ o_w.T


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

def swiglu_ffn(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """
    SwiGLU Feed-Forward Network.
    Formula: down( silu(gate(x)) * up(x) )
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    gate_pre = x @ gate_w.T                                # (seq, ff_dim)
    gate     = gate_pre / (1.0 + np.exp(-gate_pre))        # SiLU
    up       = x @ up_w.T                                  # (seq, ff_dim)
    return (gate * up) @ down_w.T                          # (seq, hidden)
