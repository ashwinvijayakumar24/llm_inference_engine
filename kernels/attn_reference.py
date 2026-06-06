"""
Torch reference + Python wrappers for the decode-attention CUDA kernel.

The reference is the gold standard the kernel is diffed against. The wrappers
pass torch CUDA tensor device pointers (.data_ptr()) into the compiled kernel —
data stays on the GPU.
"""

import torch


def attention_decode_reference(
    q: torch.Tensor,   # (n_heads, head_dim)
    k: torch.Tensor,   # (n_kv_heads, kv_seq, head_dim)
    v: torch.Tensor,   # (n_kv_heads, kv_seq, head_dim)
    scale: float,
) -> torch.Tensor:
    """
    Decode attention for one query token. Returns (n_heads, head_dim).
    fp32 math, fp16 out — mirrors the kernel's precision policy.
    GQA: query head h reads KV head h // groups.
    """
    n_heads  = q.shape[0]
    n_kv     = k.shape[0]
    kv_seq   = k.shape[1]
    head_dim = q.shape[2] if q.dim() == 3 else q.shape[1]
    groups   = n_heads // n_kv

    qf = q.float()
    kf = k.float()
    vf = v.float()

    out = torch.empty(n_heads, head_dim, dtype=torch.float32, device=q.device)
    for h in range(n_heads):
        kvh    = h // groups
        scores = (kf[kvh] @ qf[h]) * scale          # (kv_seq,)
        p      = torch.softmax(scores, dim=0)        # (kv_seq,)
        out[h] = p @ vf[kvh]                         # (head_dim,)
    return out.half()


def _check_inputs(q, k, v):
    assert q.is_cuda and k.is_cuda and v.is_cuda, "tensors must be on CUDA"
    assert q.dtype == torch.float16, "q must be fp16"
    assert k.dtype == torch.float16 and v.dtype == torch.float16, "k,v must be fp16"
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    return q, k, v


def attention_decode(q, k, v, scale, version="v1"):
    """
    Call a CUDA decode-attention kernel. Shapes:
        q: (n_heads, head_dim)  k,v: (n_kv_heads, kv_seq, head_dim)
    Returns out (n_heads, head_dim) fp16 on the same device.
    version: "v1" | "v2" | "v3"
    """
    import engine_kernels

    q, k, v = _check_inputs(q, k, v)
    n_heads, head_dim = q.shape
    n_kv, kv_seq, _   = k.shape
    out = torch.empty(n_heads, head_dim, dtype=torch.float16, device=q.device)

    fn = getattr(engine_kernels, f"attention_decode_{version}")
    fn(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        n_heads, n_kv, kv_seq, head_dim, float(scale),
    )
    return out


# Back-compat alias.
def attention_decode_v1(q, k, v, scale):
    return attention_decode(q, k, v, scale, version="v1")
