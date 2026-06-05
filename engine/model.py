"""
Forward pass wiring: embed → N×transformer_block → final_norm → lm_head.
Greedy decode loop (no KV cache — Phase 1).
"""

import numpy as np

from engine.components import (
    apply_rope,
    gqa_attention,
    precompute_rope_tables,
    rms_norm,
    swiglu_ffn,
)

EOS_IDS = {128001, 128008, 128009}


class LlamaModel:
    """
    Llama 3.2 1B forward pass in NumPy.

    Pre-norm residual layout (standard Llama):
        x = embed(token_ids)
        for each layer:
            h = rms_norm(x, input_layernorm)
            h = gqa_attention(h, ...)
            x = x + h                       # residual 1
            h = rms_norm(x, post_attn_norm)
            h = swiglu_ffn(h, ...)
            x = x + h                       # residual 2
        x = rms_norm(x, final_norm)
        logits = x @ embed_weight.T         # tied lm_head
    """

    def __init__(self, weights: dict, config: dict):
        self.weights  = weights
        self.config   = config
        self.n_heads  = config["num_attention_heads"]
        self.n_kv     = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.n_layers = config["num_hidden_layers"]
        self.eps      = config["rms_norm_eps"]

        # Precompute RoPE tables once — reused every forward call
        self.cos, self.sin = precompute_rope_tables(
            max_seq     = config["max_position_embeddings"],
            head_dim    = config["head_dim"],
            theta       = config["rope_theta"],
            rope_scaling= config.get("rope_scaling"),
        )

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """
        Run full forward pass.

        Args:
            token_ids: list of int token IDs (including BOS)

        Returns:
            logits: np.ndarray of shape (seq, vocab_size)
        """
        w   = self.weights
        seq = len(token_ids)

        x         = w["model.embed_tokens.weight"][token_ids]   # (seq, hidden)
        positions = np.arange(seq, dtype=np.int32)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"

            # --- Attention sublayer ---
            h = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
            )
            x = x + h   # residual 1

            # --- FFN sublayer ---
            h = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn(
                h,
                w[f"{p}.mlp.gate_proj.weight"],
                w[f"{p}.mlp.up_proj.weight"],
                w[f"{p}.mlp.down_proj.weight"],
            )
            x = x + h   # residual 2

        x      = rms_norm(x, w["model.norm.weight"], self.eps)
        logits = x @ w["lm_head.weight"].T   # (seq, vocab)
        return logits

    def forward_debug(self, token_ids: list[int]) -> tuple[np.ndarray, dict]:
        """
        Same as forward() but also returns intermediate captures.
        Capture keys match oracle.py fixture keys for direct comparison.

        Returns:
            (logits, captures) where captures is a dict with:
                post_embed, layer_i_post_attn, layer_i_post_ffn,
                post_final_norm
        """
        w        = self.weights
        seq      = len(token_ids)
        captures = {}

        x         = w["model.embed_tokens.weight"][token_ids]
        positions = np.arange(seq, dtype=np.int32)

        captures["post_embed"] = x.copy()

        for i in range(self.n_layers):
            p = f"model.layers.{i}"

            h = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
            )
            x = x + h
            captures[f"layer_{i}_post_attn"] = x.copy()

            h = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn(
                h,
                w[f"{p}.mlp.gate_proj.weight"],
                w[f"{p}.mlp.up_proj.weight"],
                w[f"{p}.mlp.down_proj.weight"],
            )
            x = x + h
            captures[f"layer_{i}_post_ffn"] = x.copy()

        x      = rms_norm(x, w["model.norm.weight"], self.eps)
        captures["post_final_norm"] = x.copy()

        logits = x @ w["lm_head.weight"].T
        return logits, captures


    def prefill(self, token_ids: list[int], kv_cache) -> np.ndarray:
        """
        Run forward pass for the prompt, writing all K/V to cache.
        Returns logits for the last token only — shape (vocab,).
        """
        w   = self.weights
        seq = len(token_ids)

        x         = w["model.embed_tokens.weight"][token_ids]
        positions = np.arange(seq, dtype=np.int32)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
                kv_cache=kv_cache, layer_idx=i,
            )
            x = x + h
            h = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn(
                h,
                w[f"{p}.mlp.gate_proj.weight"],
                w[f"{p}.mlp.up_proj.weight"],
                w[f"{p}.mlp.down_proj.weight"],
            )
            x = x + h

        kv_cache.advance(seq)
        last = rms_norm(x[-1:], w["model.norm.weight"], self.eps)  # (1, hidden)
        return (last @ w["lm_head.weight"].T)[0]                    # (vocab,)

    def decode_step(self, token_id: int, kv_cache) -> np.ndarray:
        """
        Run one decode step for a single new token.
        Reads full K/V history from cache, writes new K/V, advances pos.
        Returns logits — shape (vocab,).
        """
        w = self.weights

        x         = w["model.embed_tokens.weight"][[token_id]]        # (1, hidden)
        positions = np.array([kv_cache.pos], dtype=np.int32)

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, positions,
                self.n_heads, self.n_kv, self.head_dim,
                kv_cache=kv_cache, layer_idx=i,
            )
            x = x + h
            h = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn(
                h,
                w[f"{p}.mlp.gate_proj.weight"],
                w[f"{p}.mlp.up_proj.weight"],
                w[f"{p}.mlp.down_proj.weight"],
            )
            x = x + h

        kv_cache.advance(1)
        x = rms_norm(x, w["model.norm.weight"], self.eps)
        return (x @ w["lm_head.weight"].T)[0]   # (vocab,)


def greedy_decode(
    model: LlamaModel,
    token_ids: list[int],
    max_tokens: int,
    eos_ids: set[int] = EOS_IDS,
) -> list[int]:
    """
    Generate tokens greedily (no KV cache — full prefill each step).
    Returns only the newly generated token IDs (not the prompt).
    """
    ids = list(token_ids)
    for _ in range(max_tokens):
        logits  = model.forward(ids)
        next_id = int(np.argmax(logits[-1]))
        ids.append(next_id)
        if next_id in eos_ids:
            break
    return ids[len(token_ids):]
