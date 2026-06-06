"""
GPU forward pass for Llama 3.2 — torch fp16 on cuda:0 (Phase 3.2).
Same prefill() / decode_step() interface as LlamaModel.
Returns CPU numpy arrays so the existing greedy sampler works unchanged.
"""

import numpy as np
import torch

from engine.cache import KVCacheGPU
from engine.components_gpu import (
    apply_rope_gpu,
    gqa_attention_gpu,
    precompute_rope_tables_gpu,
    rms_norm_gpu,
    swiglu_ffn_gpu,
)

EOS_IDS = {128001, 128008, 128009}


class LlamaModelGPU:
    def __init__(self, weights: dict, config: dict, device: str = "cuda:0",
                 use_cuda_attn: bool = False, cuda_attn_version: str = "v3"):
        self.weights  = weights
        self.config   = config
        self.device   = device
        self.n_heads  = config["num_attention_heads"]
        self.n_kv     = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.n_layers = config["num_hidden_layers"]
        self.eps      = config["rms_norm_eps"]

        self.cos, self.sin = precompute_rope_tables_gpu(
            max_seq      = config["max_position_embeddings"],
            head_dim     = config["head_dim"],
            theta        = config["rope_theta"],
            rope_scaling = config.get("rope_scaling"),
            device       = device,
        )

        # Optional custom CUDA decode kernel. Set up the callable once here so the
        # decode loop just passes it through. None => PyTorch decode path.
        self._decode_kernel = None
        if use_cuda_attn:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(root / "build"))
            sys.path.insert(0, str(root / "kernels"))
            from attn_reference import attention_decode
            ver = cuda_attn_version
            self._decode_kernel = lambda q, k, v, scale: attention_decode(q, k, v, scale, version=ver)

    def make_cache(self, max_seq: int = 2048) -> KVCacheGPU:
        return KVCacheGPU(self.n_layers, max_seq, self.n_kv, self.head_dim, self.device)

    def prefill(self, token_ids: list[int], kv_cache: KVCacheGPU) -> np.ndarray:
        """Process prompt, write K/V to cache. Returns logits (vocab,) as CPU numpy."""
        w   = self.weights
        seq = len(token_ids)

        ids_t     = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        x         = w["model.embed_tokens.weight"][ids_t]               # (seq, hidden) fp16
        positions = torch.arange(seq, dtype=torch.long, device=self.device)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm_gpu(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention_gpu(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
                kv_cache=kv_cache, layer_idx=i,
                decode_kernel=self._decode_kernel,
            )
            x = x + h
            h = rms_norm_gpu(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn_gpu(h, w[f"{p}.mlp.gate_proj.weight"],
                               w[f"{p}.mlp.up_proj.weight"], w[f"{p}.mlp.down_proj.weight"])
            x = x + h

        kv_cache.advance(seq)
        last   = rms_norm_gpu(x[-1:], w["model.norm.weight"], self.eps)
        logits = (last @ w["lm_head.weight"].T)[0]   # (vocab,) fp16
        return logits.cpu().float().numpy()

    def forward_all(self, token_ids: list[int]) -> np.ndarray:
        """
        No-cache forward over the full sequence, returning logits at EVERY
        position — shape (seq, vocab). Used by perplexity eval (teacher forcing).
        Not on the hot path; O(seq^2) attention is fine for offline eval.
        """
        w   = self.weights
        seq = len(token_ids)

        ids_t     = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        x         = w["model.embed_tokens.weight"][ids_t]
        positions = torch.arange(seq, dtype=torch.long, device=self.device)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm_gpu(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention_gpu(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
            )
            x = x + h
            h = rms_norm_gpu(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn_gpu(h, w[f"{p}.mlp.gate_proj.weight"],
                               w[f"{p}.mlp.up_proj.weight"], w[f"{p}.mlp.down_proj.weight"])
            x = x + h

        x      = rms_norm_gpu(x, w["model.norm.weight"], self.eps)
        logits = x @ w["lm_head.weight"].T   # (seq, vocab) fp16
        return logits.cpu().float().numpy()

    def decode_step(self, token_id: int, kv_cache: KVCacheGPU) -> np.ndarray:
        """One decode step. Returns logits (vocab,) as CPU numpy."""
        w = self.weights

        ids_t     = torch.tensor([token_id], dtype=torch.long, device=self.device)
        x         = w["model.embed_tokens.weight"][ids_t]               # (1, hidden) fp16
        positions = torch.tensor([kv_cache.pos], dtype=torch.long, device=self.device)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm_gpu(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention_gpu(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
                kv_cache=kv_cache, layer_idx=i,
                decode_kernel=self._decode_kernel,
            )
            x = x + h
            h = rms_norm_gpu(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn_gpu(h, w[f"{p}.mlp.gate_proj.weight"],
                               w[f"{p}.mlp.up_proj.weight"], w[f"{p}.mlp.down_proj.weight"])
            x = x + h

        kv_cache.advance(1)
        x      = rms_norm_gpu(x, w["model.norm.weight"], self.eps)
        logits = (x @ w["lm_head.weight"].T)[0]   # (vocab,) fp16
        return logits.cpu().float().numpy()
