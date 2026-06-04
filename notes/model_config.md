# Llama 3.2 1B Instruct — Model Config

Source: `weights/config.json` (read 2026-06-04)
HF repo: `meta-llama/Llama-3.2-1B-Instruct`

## Core Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 2048 | embedding dim / d_model |
| `num_attention_heads` | 32 | query heads |
| `num_key_value_heads` | 8 | KV heads (GQA) |
| `head_dim` | 64 | explicitly set; = hidden_size / num_attention_heads = 2048/32 ✓ |
| `num_hidden_layers` | 16 | transformer blocks |
| `intermediate_size` | 8192 | FFN hidden dim (4× hidden_size) |
| `vocab_size` | 128256 | token vocabulary size |
| `max_position_embeddings` | 131072 | max sequence length (with RoPE scaling) |
| `rope_theta` | 500000.0 | RoPE base frequency |
| `rms_norm_eps` | 1e-05 | epsilon inside RMSNorm sqrt |
| `tie_word_embeddings` | true | lm_head weight == embed weight (no separate lm_head tensor) |
| `torch_dtype` | bfloat16 | native weight dtype |
| `attention_bias` | false | no bias on Q/K/V/O projections |
| `mlp_bias` | false | no bias on gate/up/down projections |

## Special Token IDs

| Token | ID |
|-------|-----|
| BOS | 128000 |
| EOS | 128001, 128008, 128009 (multiple valid EOS tokens) |

## RoPE Scaling — IMPORTANT ⚠️

This model uses **Llama 3 scaled RoPE**, NOT standard RoPE. Standard `theta=500000` alone is insufficient.

```json
"rope_scaling": {
  "rope_type": "llama3",
  "factor": 32.0,
  "high_freq_factor": 4.0,
  "low_freq_factor": 1.0,
  "original_max_position_embeddings": 8192
}
```

The `llama3` rope type applies frequency-dependent scaling: low-frequency components are scaled by `factor` (32×), high-frequency components are unscaled, with a smooth interpolation between. This extends context from 8192 → 131072 tokens.

**Implementation note for Phase 1:** Must implement `llama3` RoPE scaling in `components.py`, not plain RoPE. The HF source for reference: `transformers/models/llama/modeling_llama.py` → `LlamaRotaryEmbedding` with `rope_type="llama3"`.

## GQA Configuration

- Group ratio: `num_attention_heads / num_key_value_heads` = 32 / 8 = **4**
- Each KV head is shared by 4 query heads
- During attention: K and V tensors are repeated 4× to match Q head count

## Derived Tensor Shapes (per layer)

| Tensor | Shape | Derivation |
|--------|-------|-----------|
| `q_proj.weight` | (2048, 2048) | (n_heads × head_dim, hidden_size) |
| `k_proj.weight` | (512, 2048) | (n_kv_heads × head_dim, hidden_size) |
| `v_proj.weight` | (512, 2048) | (n_kv_heads × head_dim, hidden_size) |
| `o_proj.weight` | (2048, 2048) | (hidden_size, n_heads × head_dim) |
| `gate_proj.weight` | (8192, 2048) | (intermediate_size, hidden_size) |
| `up_proj.weight` | (8192, 2048) | (intermediate_size, hidden_size) |
| `down_proj.weight` | (2048, 8192) | (hidden_size, intermediate_size) |
| `input_layernorm.weight` | (2048,) | (hidden_size,) |
| `post_attention_layernorm.weight` | (2048,) | (hidden_size,) |

## Global Tensors

| Tensor | Shape | Notes |
|--------|-------|-------|
| `model.embed_tokens.weight` | (128256, 2048) | (vocab_size, hidden_size) |
| `model.norm.weight` | (2048,) | final RMSNorm |
| `lm_head.weight` | **absent** | tied to embed_tokens.weight |

## KV Cache Sizing Reference

Per layer, per sequence:
- K cache shape: `(max_seq, n_kv_heads, head_dim)` = `(max_seq, 8, 64)`
- V cache shape: `(max_seq, n_kv_heads, head_dim)` = `(max_seq, 8, 64)`
- Total KV per layer per token: `2 × 8 × 64 × dtype_bytes`
- At bfloat16 (2 bytes), 1024-token sequence: `2 × 8 × 64 × 2 × 1024 = 2 MB per layer`
- All 16 layers, 1024 tokens: ~32 MB KV cache
