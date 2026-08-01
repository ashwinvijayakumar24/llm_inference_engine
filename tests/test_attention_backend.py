"""
BatchMeta structural invariants and the AttentionBackend seam.

Pure CPU, no weights, no CUDA — runs in CI on every push. That is deliberate:
the metadata layer is where off-by-one errors live, and every one of them is
SILENT. A wrong kv_last_page_len or a mis-derived cu_query_lens does not crash;
it makes attention read the wrong keys and produce fluent, wrong text.

    pytest tests/test_attention_backend.py -v
"""

import pytest
import torch

from engine.attention_backend import AttentionBackend, BatchMeta

PAGE = 16


def make_meta(kv_lens, query_lens, page_size=PAGE, **overrides):
    """
    Build a well-formed BatchMeta from per-sequence (kv_len, query_len).

    This is the reference construction — the thing a scheduler's batch assembly
    must reproduce. Written here once so the tests below can corrupt one field
    at a time and confirm validate() notices.
    """
    n = len(kv_lens)
    assert len(query_lens) == n

    cu = [0]
    for q in query_lens:
        cu.append(cu[-1] + q)
    n_tokens = cu[-1]

    positions, batch_indices = [], []
    for i, (kv, q) in enumerate(zip(kv_lens, query_lens)):
        # This sequence's new tokens end at absolute position kv-1.
        start = kv - q
        positions.extend(range(start, kv))
        batch_indices.extend([i] * q)

    # CSR page addressing. Page ids are allocated sequentially here; a real
    # allocator hands back arbitrary free-list ids, which is exactly why the
    # indirection exists.
    kv_indptr, kv_indices, last_page = [0], [], []
    next_page = 0
    for kv in kv_lens:
        n_pages = (kv + page_size - 1) // page_size
        kv_indices.extend(range(next_page, next_page + n_pages))
        next_page += n_pages
        kv_indptr.append(len(kv_indices))
        # 1 <= last <= page_size. An exact multiple reports page_size, NOT 0.
        last_page.append(kv - (n_pages - 1) * page_size)

    slot_mapping = []
    for i, (kv, q) in enumerate(zip(kv_lens, query_lens)):
        pages = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]
        for p in range(kv - q, kv):
            slot_mapping.append(pages[p // page_size] * page_size + (p % page_size))

    last_token_ix = [cu[i + 1] - 1 for i in range(n)]

    def t(x, dtype=torch.int32):
        return torch.tensor(x, dtype=dtype)

    fields = dict(
        query_lens=t(query_lens),
        cu_query_lens=t(cu),
        kv_lens=t(kv_lens),
        positions=t(positions, torch.int64),
        last_token_ix=t(last_token_ix),
        kv_indptr=t(kv_indptr),
        kv_indices=t(kv_indices),
        kv_last_page_len=t(last_page),
        batch_indices=t(batch_indices),
        slot_mapping=t(slot_mapping),
        page_size=page_size,
        is_prefill=any(q > 1 for q in query_lens),
    )
    fields.update(overrides)
    return BatchMeta(**fields)


# --------------------------------------------------------------------------
# Well-formed cases
# --------------------------------------------------------------------------


def test_single_decode_step():
    """One sequence, one new token — the simplest batch there is."""
    meta = make_meta(kv_lens=[10], query_lens=[1])
    meta.validate()
    assert meta.n_seqs == 1
    assert meta.n_tokens == 1
    assert not meta.is_prefill


def test_mixed_prefill_and_decode():
    """
    A chunked prefill batched with two decodes — the shape continuous batching
    exists to produce, and the reason cu_query_lens is in the contract at all.
    """
    meta = make_meta(kv_lens=[40, 7, 100], query_lens=[8, 1, 1])
    meta.validate()
    assert meta.n_seqs == 3
    assert meta.n_tokens == 10
    assert meta.is_prefill
    assert meta.cu_query_lens.tolist() == [0, 8, 9, 10]
    # Each sequence's logits come from its final token.
    assert meta.last_token_ix.tolist() == [7, 8, 9]


@pytest.mark.parametrize("kv_len", [1, 15, 16, 17, 31, 32, 33, 64, 65])
def test_page_boundary_lengths(kv_len):
    """
    Sweep lengths across page boundaries.

    THE case this guards: kv_len an exact multiple of page_size must report
    kv_last_page_len == page_size, never 0. Off by one page of attention is
    silent — no crash, plausible output, wrong result.
    """
    meta = make_meta(kv_lens=[kv_len], query_lens=[1])
    meta.validate()

    expected_pages = (kv_len + PAGE - 1) // PAGE
    assert len(meta.kv_indices) == expected_pages

    last = int(meta.kv_last_page_len[0])
    assert 1 <= last <= PAGE
    if kv_len % PAGE == 0:
        assert last == PAGE, f"kv_len={kv_len} is a multiple of {PAGE}; last page must be {PAGE}, not 0"
    else:
        assert last == kv_len % PAGE


def test_positions_are_absolute_not_relative():
    """
    positions must be each token's ABSOLUTE index in its own sequence.

    This is the whole reason positions moved into BatchMeta. A chunk starting at
    absolute position 32 must carry 32.., not 0.. — the bug that existed in
    model_gpu.prefill and was invisible because prefill ran only once per request.
    """
    meta = make_meta(kv_lens=[40], query_lens=[8])
    meta.validate()
    assert meta.positions.tolist() == list(range(32, 40))


def test_slot_mapping_matches_csr():
    """slot_mapping and the CSR triple must describe the same physical storage."""
    meta = make_meta(kv_lens=[20], query_lens=[4])
    meta.validate()
    pages = meta.kv_indices.tolist()
    for j, pos in enumerate(meta.positions.tolist()):
        expected = pages[pos // PAGE] * PAGE + (pos % PAGE)
        assert int(meta.slot_mapping[j]) == expected, f"token {j} at pos {pos}"


# --------------------------------------------------------------------------
# Corruption cases — validate() must catch each one
# --------------------------------------------------------------------------


def test_rejects_zero_last_page_len():
    """The off-by-one that motivates the invariant."""
    meta = make_meta(kv_lens=[32], query_lens=[1],
                     kv_last_page_len=torch.tensor([0], dtype=torch.int32))
    with pytest.raises(AssertionError, match=r"kv_last_page_len must lie in"):
        meta.validate()


def test_rejects_last_page_len_over_page_size():
    meta = make_meta(kv_lens=[20], query_lens=[1],
                     kv_last_page_len=torch.tensor([PAGE + 1], dtype=torch.int32))
    with pytest.raises(AssertionError, match=r"kv_last_page_len must lie in"):
        meta.validate()


def test_rejects_bad_cu_query_lens():
    """cu_query_lens must be the exclusive prefix sum — not merely monotonic."""
    meta = make_meta(kv_lens=[10, 10], query_lens=[1, 1],
                     cu_query_lens=torch.tensor([0, 1, 3], dtype=torch.int32))
    with pytest.raises(AssertionError, match=r"cu_query_lens\[-1\]"):
        meta.validate()


def test_rejects_page_count_mismatch():
    """Allocated pages must actually cover the claimed kv_len."""
    meta = make_meta(kv_lens=[100], query_lens=[1],
                     kv_indices=torch.tensor([0, 1], dtype=torch.int32),
                     kv_indptr=torch.tensor([0, 2], dtype=torch.int32))
    with pytest.raises(AssertionError, match=r"pages allocated but kv_len"):
        meta.validate()


def test_rejects_batch_indices_length_mismatch():
    meta = make_meta(kv_lens=[10, 10], query_lens=[1, 1],
                     batch_indices=torch.tensor([0], dtype=torch.int32))
    with pytest.raises(AssertionError, match=r"batch_indices"):
        meta.validate()


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


def test_backend_protocol_is_structural():
    """
    AttentionBackend is runtime_checkable so a serving layer can reject a
    malformed plugin at startup rather than mid-benchmark.

    Note the limitation, which is why this is a startup smoke check and not a
    correctness guarantee: Protocol runtime checks verify method PRESENCE only,
    never signatures. A backend with the right names and wrong arguments passes.
    """

    class Complete:
        def append_kv(self, layer_idx, k, v, meta): ...
        def attend(self, q, layer_idx, scale, meta): ...

    class MissingAttend:
        def append_kv(self, layer_idx, k, v, meta): ...

    assert isinstance(Complete(), AttentionBackend)
    assert not isinstance(MissingAttend(), AttentionBackend)


def test_meta_is_immutable():
    """
    Frozen because one BatchMeta describes one forward pass and is read by all
    16 layers. A backend mutating it mid-pass would desynchronise later layers
    from earlier ones.
    """
    meta = make_meta(kv_lens=[10], query_lens=[1])
    with pytest.raises((AttributeError, TypeError)):
        meta.page_size = 32
