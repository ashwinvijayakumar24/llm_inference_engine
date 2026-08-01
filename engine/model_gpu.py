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

        # RoPE positions continue from wherever the cache already is, rather than
        # restarting at 0. Today this is a no-op — generate() builds a fresh cache
        # and calls prefill exactly once (engine/scheduler.py:26-33), so pos is
        # always 0 and arange(0, seq) == arange(seq). It matters the moment a
        # prompt is prefilled in more than one chunk: the second chunk would
        # otherwise be told it starts at position 0, silently corrupting RoPE for
        # every token in it. Latent bug, fixed before anything depends on it.
        positions = torch.arange(
            kv_cache.pos, kv_cache.pos + seq, dtype=torch.long, device=self.device
        )

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

    def forward_varlen(self, token_ids, meta, backend) -> "torch.Tensor":
        """
        Batched forward over a ragged batch of sequences. The serving path.

        Deliberately a SIBLING of prefill()/decode_step() rather than a
        replacement. Those two keep their exact behaviour, so the engine's
        existing correctness tests and published batch-1 benchmarks continue to
        describe live code rather than history.

        Args:
            token_ids: (tokens,) int64 tensor ON DEVICE. All sequences' new tokens
                       packed along one axis in sequence order — see BatchMeta.
            meta:      BatchMeta describing the batch (engine/attention_backend.py).
            backend:   AttentionBackend owning KV storage and attention.

        Returns:
            (n_seqs, vocab) fp16 logits, ON DEVICE — one row per sequence, taken
            at that sequence's last token.

        WHY THIS RETURNS A DEVICE TENSOR WHEN prefill() RETURNS CPU NUMPY
        ----------------------------------------------------------------
        prefill/decode_step end with `.cpu().float().numpy()` so the NumPy
        sampler works unchanged — a deliberate engine choice costing ~0.1 ms
        against a ~12 ms step. At batch 32 that same round-trip sits on the
        critical path of every request in the batch, and sampling belongs on the
        GPU anyway.

        CONSEQUENCE THE CALLER MUST KNOW: that per-token device->host copy is
        what currently forces a CUDA sync and makes host-clock timing of this
        engine meaningful. Without it, `time.perf_counter()` around this call
        measures kernel-launch queueing, not execution — timings get FASTER and
        nothing errors. Any serving layer timing this path must use CUDA events
        or an explicit, declared sync point.

        WHY MOST OF THE FORWARD PASS IS IDENTICAL TO prefill()
        -----------------------------------------------------
        Compare this loop to prefill()'s: the only differences are that positions
        arrive in `meta`, attention goes through `backend`, and the final logits
        are a gather instead of a slice. Every linear, norm and FFN is untouched,
        because a flattened varlen layout makes them unable to tell that more
        than one sequence is present. That is the entire argument for varlen.
        """
        w = self.weights
        tokens = int(token_ids.shape[0])

        x = w["model.embed_tokens.weight"][token_ids]        # (tokens, hidden) fp16

        for i in range(self.n_layers):
            p = f"model.layers.{i}"
            h = rms_norm_gpu(x, w[f"{p}.input_layernorm.weight"], self.eps)
            h = gqa_attention_gpu(
                h,
                w[f"{p}.self_attn.q_proj.weight"],
                w[f"{p}.self_attn.k_proj.weight"],
                w[f"{p}.self_attn.v_proj.weight"],
                w[f"{p}.self_attn.o_proj.weight"],
                self.cos, self.sin, meta.positions,
                self.n_heads, self.n_kv, self.head_dim,
                backend=backend, layer_idx=i, meta=meta,
            )
            x = x + h
            h = rms_norm_gpu(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
            h = swiglu_ffn_gpu(h, w[f"{p}.mlp.gate_proj.weight"],
                               w[f"{p}.mlp.up_proj.weight"], w[f"{p}.mlp.down_proj.weight"])
            x = x + h

        # Gather each sequence's last token BEFORE the lm_head projection.
        # lm_head is (hidden, vocab) with vocab = 128256, so projecting all
        # `tokens` rows and discarding most is the single most expensive
        # avoidable operation in this function. During a chunked prefill nearly
        # every row is discarded.
        last = x[meta.last_token_ix.long()]                   # (n_seqs, hidden)
        last = rms_norm_gpu(last, w["model.norm.weight"], self.eps)
        return last @ w["lm_head.weight"].T                   # (n_seqs, vocab) fp16

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
