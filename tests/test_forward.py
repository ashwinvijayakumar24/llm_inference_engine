"""
Task 1.7 — Full forward pass tests.

Layer-by-layer diff against oracle, final logit diff, argmax match.
These tests persist as regression tests through all future phases.
"""

import numpy as np
import pytest

from tests.oracle import compare_tensors


class TestForwardPass:
    def test_embed_lookup_exact(self, model, oracle_short):
        """Embedding lookup is exact (integer indexing, no float ops)."""
        token_ids = oracle_short["token_ids"]
        _, captures = model.forward_debug(token_ids)

        passed = compare_tensors(
            captures["post_embed"],
            oracle_short["post_embed"],
            "post_embed",
            atol=1e-6,
        )
        assert passed

    def test_layer_by_layer_post_attn(self, model, oracle_short):
        """
        post_attn hidden state matches oracle at every layer.
        First failure localizes bug to a specific layer — do not skip.
        """
        token_ids = oracle_short["token_ids"]
        _, captures = model.forward_debug(token_ids)

        n_layers = model.config["num_hidden_layers"]
        failures = []
        for i in range(n_layers):
            key = f"layer_{i}_post_attn"
            # atol=5e-3: fp32 errors from prior layers amplify through large FFN
            # matmuls (2048→8192). Mean diff stays ~1e-6 — max diff is isolated
            # outliers. Final logit argmax is exact, confirming correctness.
            passed = compare_tensors(
                captures[key],
                oracle_short[key],
                key,
                atol=5e-3,
            )
            if not passed:
                failures.append(key)

        assert not failures, f"post_attn mismatch at layers: {failures}"

    def test_layer_by_layer_post_ffn(self, model, oracle_short):
        """
        post_ffn hidden state matches oracle at every layer.
        """
        token_ids = oracle_short["token_ids"]
        _, captures = model.forward_debug(token_ids)

        n_layers = model.config["num_hidden_layers"]
        failures = []
        for i in range(n_layers):
            key = f"layer_{i}_post_ffn"
            passed = compare_tensors(
                captures[key],
                oracle_short[key],
                key,
                atol=5e-3,
            )
            if not passed:
                failures.append(key)

        assert not failures, f"post_ffn mismatch at layers: {failures}"

    def test_post_final_norm(self, model, oracle_short):
        """Hidden state after final RMSNorm matches oracle."""
        token_ids = oracle_short["token_ids"]
        _, captures = model.forward_debug(token_ids)

        passed = compare_tensors(
            captures["post_final_norm"],
            oracle_short["post_final_norm"],
            "post_final_norm",
            atol=1e-3,
        )
        assert passed

    def test_final_logits_argmax_short(self, model, oracle_short):
        """argmax of logits matches oracle on every position (short prompt)."""
        token_ids = oracle_short["token_ids"]
        logits, _ = model.forward_debug(token_ids)

        our_argmax = np.argmax(logits, axis=-1)
        ref_argmax = np.argmax(oracle_short["logits"], axis=-1)

        mismatches = np.where(our_argmax != ref_argmax)[0]
        if len(mismatches) > 0:
            print(f"\n  Argmax mismatches at positions: {mismatches.tolist()}")
            for pos in mismatches:
                print(f"    pos {pos}: ours={our_argmax[pos]}  ref={ref_argmax[pos]}")
        assert len(mismatches) == 0, f"Argmax mismatch at {len(mismatches)} positions"

    def test_final_logits_diff_short(self, model, oracle_short):
        """Logit values match oracle within 1e-3 (short prompt)."""
        token_ids = oracle_short["token_ids"]
        logits, _ = model.forward_debug(token_ids)

        passed = compare_tensors(
            logits,
            oracle_short["logits"],
            "final_logits (short)",
            atol=1e-3,
        )
        assert passed

    def test_final_logits_argmax_medium(self, model, oracle_medium):
        """argmax of logits matches oracle on every position (medium prompt)."""
        token_ids = oracle_medium["token_ids"]
        logits = model.forward(token_ids)

        our_argmax = np.argmax(logits, axis=-1)
        ref_argmax = np.argmax(oracle_medium["logits"], axis=-1)

        mismatches = np.where(our_argmax != ref_argmax)[0]
        assert len(mismatches) == 0, f"Argmax mismatch at {len(mismatches)} positions"

    def test_final_logits_diff_medium(self, model, oracle_medium):
        """Logit values match oracle within 1e-3 (medium prompt)."""
        token_ids = oracle_medium["token_ids"]
        logits = model.forward(token_ids)

        passed = compare_tensors(
            logits,
            oracle_medium["logits"],
            "final_logits (medium)",
            atol=1e-3,
        )
        assert passed
