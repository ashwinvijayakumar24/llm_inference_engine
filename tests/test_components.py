"""
Tasks 1.3–1.6 — Component tests vs HF oracle.

Each test grabs the matching tensor from the oracle fixture and diffs
our NumPy implementation against the HF fp32 reference.
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from pathlib import Path

from engine.components import (
    apply_rope,
    gqa_attention,
    precompute_rope_tables,
    rms_norm,
    swiglu_ffn,
)
from tests.oracle import compare_tensors, WEIGHTS_DIR

# ---------------------------------------------------------------------------
# RMSNorm  (Task 1.3)
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_formula_correctness(self, weights, config):
        """Output matches hand-computed formula on random input."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, config["hidden_size"])).astype(np.float32)
        w = weights["model.layers.0.input_layernorm.weight"]
        eps = config["rms_norm_eps"]

        out = rms_norm(x, w, eps)
        # Hand-compute
        expected = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) * w
        assert np.allclose(out, expected, atol=1e-6), "Formula differs from hand-compute"

    def test_vs_hf_oracle(self, weights, config):
        """Diff vs HF LlamaRMSNorm on a real layer weight, atol=1e-5."""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, config["hidden_size"])).astype(np.float32)
        w    = weights["model.layers.0.input_layernorm.weight"]
        eps  = config["rms_norm_eps"]

        # HF reference
        hf_norm = hf_model.model.layers[0].input_layernorm
        x_th    = torch.tensor(x_np)
        with torch.no_grad():
            ref = hf_norm(x_th).float().numpy()

        out  = rms_norm(x_np, w, eps)
        passed = compare_tensors(out, ref, "rms_norm layer0", atol=1e-5)
        assert passed

    def test_small_magnitude(self, weights, config):
        """Numerically stable on very small inputs (~1e-6)."""
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((4, config["hidden_size"])) * 1e-6).astype(np.float32)
        w = weights["model.layers.0.input_layernorm.weight"]
        out = rms_norm(x, w, config["rms_norm_eps"])
        assert np.all(np.isfinite(out)), "Non-finite output on small input"

    def test_large_magnitude(self, weights, config):
        """Numerically stable on large inputs (~1e3)."""
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((4, config["hidden_size"])) * 1e3).astype(np.float32)
        w = weights["model.layers.0.input_layernorm.weight"]
        out = rms_norm(x, w, config["rms_norm_eps"])
        assert np.all(np.isfinite(out)), "Non-finite output on large input"


# ---------------------------------------------------------------------------
# RoPE  (Task 1.4)
# ---------------------------------------------------------------------------

class TestRoPE:
    def _get_hf_rope(self, hf_model, seq: int):
        """Extract cos/sin from HF model (API-version-agnostic)."""
        pos_ids = torch.arange(seq).unsqueeze(0)
        # transformers>=4.45: rotary_emb is on model.model, not on each attention layer
        rotary_emb = hf_model.model.rotary_emb
        with torch.no_grad():
            cos_hf, sin_hf = rotary_emb(
                hf_model.model.embed_tokens.weight, position_ids=pos_ids
            )
        # Returns (1, seq, head_dim) or (1, 1, seq, head_dim) — normalise to (seq, head_dim)
        cos_hf = cos_hf.squeeze().float().numpy()
        sin_hf = sin_hf.squeeze().float().numpy()
        if cos_hf.ndim == 1:  # single position edge case
            cos_hf = cos_hf[np.newaxis]
            sin_hf = sin_hf[np.newaxis]
        return cos_hf, sin_hf

    def test_cos_sin_tables_vs_hf(self, config):
        """cos/sin tables match HF LlamaRotaryEmbedding at positions 0..16."""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        seq = 17
        cos, sin = precompute_rope_tables(
            max_seq      = seq,
            head_dim     = config["head_dim"],
            theta        = config["rope_theta"],
            rope_scaling = config.get("rope_scaling"),
        )

        cos_hf, sin_hf = self._get_hf_rope(hf_model, seq)

        passed_cos = compare_tensors(cos[:seq], cos_hf, "rope cos[0:17]", atol=1e-5)
        passed_sin = compare_tensors(sin[:seq], sin_hf, "rope sin[0:17]", atol=1e-5)
        assert passed_cos and passed_sin

    def test_apply_rope_vs_hf(self, config):
        """apply_rope output matches HF on random q at positions 0..32."""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        seq = 33
        rng = np.random.default_rng(7)
        q_np = rng.standard_normal((seq, config["num_attention_heads"], config["head_dim"])).astype(np.float32)

        cos, sin = precompute_rope_tables(
            seq, config["head_dim"], config["rope_theta"], config.get("rope_scaling")
        )
        positions = np.arange(seq, dtype=np.int32)
        our_q = apply_rope(q_np, cos[positions], sin[positions])

        # HF: use apply_rotary_pos_emb with the model's own tables
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos_hf, sin_hf = self._get_hf_rope(hf_model, seq)
        # q in HF layout: (batch, n_heads, seq, head_dim)
        q_th = torch.tensor(q_np).unsqueeze(0).permute(0, 2, 1, 3)
        cos_th = torch.tensor(cos_hf).unsqueeze(0)  # (1, seq, head_dim)
        sin_th = torch.tensor(sin_hf).unsqueeze(0)
        with torch.no_grad():
            q_rot_hf, _ = apply_rotary_pos_emb(q_th, q_th, cos_th, sin_th)
        q_rot_hf = q_rot_hf[0].permute(1, 0, 2).float().numpy()  # (seq, NH, HD)

        passed = compare_tensors(our_q, q_rot_hf, "apply_rope q[0:33]", atol=1e-5)
        assert passed

    def test_position_offset(self, config):
        """RoPE at position 64 matches HF exactly."""
        from transformers import AutoModelForCausalLM
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        pos = 64
        cos, sin = precompute_rope_tables(
            128, config["head_dim"], config["rope_theta"], config.get("rope_scaling")
        )
        rng = np.random.default_rng(3)
        q_np = rng.standard_normal((1, config["num_attention_heads"], config["head_dim"])).astype(np.float32)
        our_q = apply_rope(q_np, cos[[pos]], sin[[pos]])

        cos_hf, sin_hf = self._get_hf_rope(hf_model, pos + 1)
        q_th = torch.tensor(q_np).unsqueeze(0).permute(0, 2, 1, 3)
        cos_th = torch.tensor(cos_hf[[pos]]).unsqueeze(0)
        sin_th = torch.tensor(sin_hf[[pos]]).unsqueeze(0)
        with torch.no_grad():
            q_rot_hf, _ = apply_rotary_pos_emb(q_th, q_th, cos_th, sin_th)
        q_rot_hf = q_rot_hf[0].permute(1, 0, 2).float().numpy()

        # atol=2e-5: fp32 accumulation at higher positions produces ~1.2e-5 diff — still correct
        passed = compare_tensors(our_q, q_rot_hf, f"apply_rope at pos={pos}", atol=2e-5)
        assert passed


# ---------------------------------------------------------------------------
# GQA Attention  (Task 1.5)
# ---------------------------------------------------------------------------

class TestGQAAttention:
    def test_vs_hf_layer0(self, weights, config, oracle_short):
        """
        GQA output matches HF layer-0 attention block on a real prompt.
        Uses oracle's post_embed as input (before layer norm).
        """
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        token_ids = oracle_short["token_ids"]
        seq = len(token_ids)

        # Input to layer 0 = post_embed
        x_np = oracle_short["post_embed"].astype(np.float32)  # (seq, hidden)

        # Apply input_layernorm (as done in the model forward)
        from engine.components import rms_norm as _rms_norm
        x_normed = _rms_norm(
            x_np,
            weights["model.layers.0.input_layernorm.weight"],
            config["rms_norm_eps"],
        )

        cos, sin = precompute_rope_tables(
            config["max_position_embeddings"],
            config["head_dim"],
            config["rope_theta"],
            config.get("rope_scaling"),
        )
        positions = np.arange(seq, dtype=np.int32)

        our_out = gqa_attention(
            x_normed,
            weights["model.layers.0.self_attn.q_proj.weight"],
            weights["model.layers.0.self_attn.k_proj.weight"],
            weights["model.layers.0.self_attn.v_proj.weight"],
            weights["model.layers.0.self_attn.o_proj.weight"],
            cos, sin, positions,
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
        )

        # HF reference: transformers>=4.45 requires position_embeddings tuple
        pos_ids = torch.arange(seq).unsqueeze(0)
        x_th = torch.tensor(x_normed).unsqueeze(0)  # (1, seq, hidden)
        with torch.no_grad():
            pos_embeds = hf_model.model.rotary_emb(
                hf_model.model.embed_tokens.weight, position_ids=pos_ids
            )
            hf_attn_out, _ = hf_model.model.layers[0].self_attn(
                x_th,
                position_embeddings=pos_embeds,
                attention_mask=None,
            )
        hf_out = hf_attn_out[0].float().numpy()  # (seq, hidden)

        passed = compare_tensors(our_out, hf_out, "gqa_attention layer0", atol=1e-4)
        assert passed

    def test_causal_mask(self, config):
        """Future positions contribute zero attention weight."""
        rng = np.random.default_rng(5)
        hidden = config["hidden_size"]
        seq    = 8

        # Dummy weights
        q_w = rng.standard_normal((config["num_attention_heads"] * config["head_dim"], hidden)).astype(np.float32)
        k_w = rng.standard_normal((config["num_key_value_heads"] * config["head_dim"], hidden)).astype(np.float32)
        v_w = rng.standard_normal((config["num_key_value_heads"] * config["head_dim"], hidden)).astype(np.float32)
        o_w = rng.standard_normal((hidden, config["num_attention_heads"] * config["head_dim"])).astype(np.float32)
        x   = rng.standard_normal((seq, hidden)).astype(np.float32)

        cos, sin = precompute_rope_tables(
            seq, config["head_dim"], config["rope_theta"], config.get("rope_scaling")
        )
        positions = np.arange(seq, dtype=np.int32)

        # Patch: make a second run with token at position 3 zeroed to check isolation
        out1 = gqa_attention(x, q_w, k_w, v_w, o_w, cos, sin, positions,
                             config["num_attention_heads"], config["num_key_value_heads"],
                             config["head_dim"])

        x2 = x.copy()
        x2[5] = 0   # zero out position 5
        out2 = gqa_attention(x2, q_w, k_w, v_w, o_w, cos, sin, positions,
                             config["num_attention_heads"], config["num_key_value_heads"],
                             config["head_dim"])

        # Positions 0..4 should not be affected by zeroing position 5 (causal mask)
        assert np.allclose(out1[:5], out2[:5], atol=1e-5), \
            "Causal mask violated: earlier positions affected by later token change"


# ---------------------------------------------------------------------------
# SwiGLU FFN  (Task 1.6)
# ---------------------------------------------------------------------------

class TestSwiGLUFFN:
    def test_vs_hf_layer0(self, weights, config, oracle_short):
        """SwiGLU output matches HF MLP layer-0, atol=1e-4."""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
        )
        hf_model.eval()

        # Input to MLP = post_attn state after post_attention_layernorm
        x_post_attn = oracle_short["layer_0_post_attn"].astype(np.float32)
        x_normed = rms_norm(
            x_post_attn,
            weights["model.layers.0.post_attention_layernorm.weight"],
            config["rms_norm_eps"],
        )

        our_out = swiglu_ffn(
            x_normed,
            weights["model.layers.0.mlp.gate_proj.weight"],
            weights["model.layers.0.mlp.up_proj.weight"],
            weights["model.layers.0.mlp.down_proj.weight"],
        )

        x_th = torch.tensor(x_normed).unsqueeze(0)
        with torch.no_grad():
            hf_out = hf_model.model.layers[0].mlp(x_th)
        hf_out = hf_out[0].float().numpy()

        passed = compare_tensors(our_out, hf_out, "swiglu_ffn layer0", atol=1e-4)
        assert passed

    def test_output_shape(self, weights, config):
        """Output shape matches input shape."""
        rng = np.random.default_rng(9)
        x = rng.standard_normal((7, config["hidden_size"])).astype(np.float32)
        out = swiglu_ffn(
            x,
            weights["model.layers.0.mlp.gate_proj.weight"],
            weights["model.layers.0.mlp.up_proj.weight"],
            weights["model.layers.0.mlp.down_proj.weight"],
        )
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
