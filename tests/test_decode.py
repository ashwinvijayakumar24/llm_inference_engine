"""
Task 1.8 — Greedy decode test.

32 generated token IDs must be bit-identical to HF model.generate(do_sample=False).
This is the Phase 1 end-to-end milestone.
"""

import numpy as np
import pytest

from engine.model import greedy_decode


class TestGreedyDecode:
    def test_32_tokens_match_hf_short(self, model, oracle_short):
        """
        32 greedy tokens from our engine match HF oracle exactly (short prompt).
        Token IDs must be bit-identical — not just close.
        """
        token_ids = oracle_short["token_ids"]
        ref_ids   = oracle_short["greedy_ids"]

        our_ids = greedy_decode(model, token_ids, max_tokens=len(ref_ids))

        # Trim to same length (oracle may have stopped early on EOS)
        n = min(len(our_ids), len(ref_ids))
        our_ids = our_ids[:n]
        ref_ids = ref_ids[:n]

        mismatches = [i for i, (a, b) in enumerate(zip(our_ids, ref_ids)) if a != b]
        if mismatches:
            print(f"\n  First mismatch at position {mismatches[0]}:")
            print(f"    ours: {our_ids[mismatches[0]]}")
            print(f"    ref:  {ref_ids[mismatches[0]]}")
            print(f"  ours: {our_ids}")
            print(f"  ref:  {ref_ids}")
        assert not mismatches, f"Token mismatch at positions: {mismatches}"

    def test_32_tokens_match_hf_medium(self, model, oracle_medium):
        """32 greedy tokens match HF oracle exactly (medium prompt)."""
        token_ids = oracle_medium["token_ids"]
        ref_ids   = oracle_medium["greedy_ids"]

        our_ids = greedy_decode(model, token_ids, max_tokens=len(ref_ids))

        n = min(len(our_ids), len(ref_ids))
        mismatches = [i for i, (a, b) in enumerate(zip(our_ids[:n], ref_ids[:n])) if a != b]
        assert not mismatches, f"Token mismatch at positions: {mismatches}"

    def test_stops_on_eos(self, model, oracle_short):
        """Decode stops when any EOS token is sampled."""
        from engine.model import EOS_IDS

        token_ids = oracle_short["token_ids"]
        # Generate up to 200 tokens but should stop at EOS
        our_ids = greedy_decode(model, token_ids, max_tokens=200)

        # If it stopped before 200, last token must be an EOS (or we hit the limit)
        if len(our_ids) < 200:
            assert our_ids[-1] in EOS_IDS, \
                f"Stopped early but last token {our_ids[-1]} is not EOS"
