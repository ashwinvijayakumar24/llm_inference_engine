"""From-scratch NumPy implementations: RMSNorm, RoPE, GQA attention, SwiGLU FFN."""

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    raise NotImplementedError


def precompute_rope_tables(max_seq: int, head_dim: int, theta: float):
    raise NotImplementedError


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def gqa_attention(q, k, v, mask=None):
    raise NotImplementedError


def swiglu_ffn(x: np.ndarray, gate_w, up_w, down_w) -> np.ndarray:
    raise NotImplementedError
