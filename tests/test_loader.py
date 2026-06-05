"""Task 1.2 — Weight loader tests."""

import numpy as np
import pytest


def test_all_keys_present(weights, config):
    """Every expected tensor name is in the loaded dict."""
    H   = config["hidden_size"]
    NH  = config["num_attention_heads"]
    NKV = config["num_key_value_heads"]
    HD  = config["head_dim"]
    NL  = config["num_hidden_layers"]

    assert "model.embed_tokens.weight" in weights
    assert "model.norm.weight" in weights
    assert "lm_head.weight" in weights  # alias from tied embed

    for i in range(NL):
        p = f"model.layers.{i}"
        for suffix in [
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
        ]:
            assert f"{p}{suffix}" in weights, f"Missing: {p}{suffix}"


def test_all_shapes_correct(weights, config):
    """Every tensor has the shape derived from config."""
    H   = config["hidden_size"]
    NH  = config["num_attention_heads"]
    NKV = config["num_key_value_heads"]
    HD  = config["head_dim"]
    NL  = config["num_hidden_layers"]
    FF  = config["intermediate_size"]
    V   = config["vocab_size"]

    assert weights["model.embed_tokens.weight"].shape == (V, H)
    assert weights["model.norm.weight"].shape == (H,)
    assert weights["lm_head.weight"].shape == (V, H)

    for i in range(NL):
        p = f"model.layers.{i}"
        assert weights[f"{p}.self_attn.q_proj.weight"].shape == (NH * HD, H)
        assert weights[f"{p}.self_attn.k_proj.weight"].shape == (NKV * HD, H)
        assert weights[f"{p}.self_attn.v_proj.weight"].shape == (NKV * HD, H)
        assert weights[f"{p}.self_attn.o_proj.weight"].shape == (H, NH * HD)
        assert weights[f"{p}.mlp.gate_proj.weight"].shape    == (FF, H)
        assert weights[f"{p}.mlp.up_proj.weight"].shape      == (FF, H)
        assert weights[f"{p}.mlp.down_proj.weight"].shape    == (H, FF)
        assert weights[f"{p}.input_layernorm.weight"].shape            == (H,)
        assert weights[f"{p}.post_attention_layernorm.weight"].shape   == (H,)


def test_all_dtypes_float32(weights):
    """Every tensor is fp32 (bfloat16 cast happened at load time)."""
    for name, t in weights.items():
        assert t.dtype == np.float32, f"{name}: expected float32, got {t.dtype}"


def test_tied_lm_head_is_alias(weights):
    """lm_head.weight shares memory with embed_tokens.weight (no copy)."""
    assert weights["lm_head.weight"] is weights["model.embed_tokens.weight"]
