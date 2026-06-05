"""
Torch GPU implementations of all Llama components — implement on PACE (Phase 3.2).

Mirrors engine/components.py but uses torch tensors on cuda:0 in fp16.
All function signatures should match the NumPy versions so model_gpu.py
can call them with minimal changes.
"""


def rms_norm_gpu(x, weight, eps: float):
    raise NotImplementedError


def apply_rope_gpu(x, cos, sin):
    raise NotImplementedError


def precompute_rope_tables_gpu(max_seq: int, head_dim: int, theta: float, rope_scaling=None):
    raise NotImplementedError


def gqa_attention_gpu(x, q_w, k_w, v_w, o_w, cos, sin, positions,
                      n_heads: int, n_kv_heads: int, head_dim: int,
                      kv_cache=None, layer_idx=None):
    raise NotImplementedError


def swiglu_ffn_gpu(x, gate_w, up_w, down_w):
    raise NotImplementedError
