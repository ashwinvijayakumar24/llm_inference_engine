"""
Pluggable attention/KV-cache backend — the seam between this engine and a
serving layer.

WHY THIS EXISTS
---------------
The engine's own attention path owns three things at once: it writes K/V into a
contiguous cache, reads the whole history back, and computes the attention math
(``engine/components_gpu.py:137-182``). That is correct and fast for one request,
and it is unusable for a serving layer, because:

  * the write assumes a sequence occupies a contiguous prefix ``[0, pos)`` of one
    buffer (``components_gpu.py:139-142``) — not merely contiguous memory, but
    contiguous *and starting at zero*, which is what makes multi-sequence sharing
    impossible;
  * ``pos`` is a single int on the cache object (``engine/cache.py:17,30``), so
    there is exactly one sequence in the world;
  * the custom CUDA kernel's nanobind ABI takes no strides, no block table and no
    block size (``kernels/bindings.cpp:29-31``), so a paged cache cannot even be
    *described* across that boundary.

Rather than paging the CUDA kernel — a rewrite, not an extension, since ``Q`` is
``[n_heads, head_dim]`` with no batch axis (``kernels/attention_decode.cu:30-32``)
— this module widens the injection hook that already exists.

``gqa_attention_gpu`` already accepts ``decode_kernel=`` (``components_gpu.py:114``)
and threads it from ``LlamaModelGPU.__init__`` (``model_gpu.py:44-53``). Today that
hook injects only the *math*: ``(q, k, v, scale) -> out``. The engine keeps owning
the cache. An ``AttentionBackend`` owns *both* — the append and the attend — so a
serving layer can supply a paged, batched implementation without this repo
learning what a block is.

WHAT THIS BUYS, CONCRETELY
--------------------------
The engine's existing contiguous path is untouched and its published numbers
stand. A serving layer injects a backend and gets paged, batched attention with
no fork. The custom CUDA kernel remains the single-request contiguous reference
path — it is NOT used on a paged path, and that is a cost worth stating out loud
rather than discovering under questioning.

WHY VARLEN, NOT PADDED
----------------------
Tokens from every sequence are packed along ONE axis::

    3 sequences decoding:        [ a0 | b0 | c0 ]            -> (3, hidden)
    1 prefill chunk + 2 decodes: [ d0 d1 d2 d3 | a0 | b0 ]   -> (6, hidden)

This is the choice that makes batching cheap. Under this layout every
position-independent operation in the forward pass is *literally unchanged*:
``linear`` (``components_gpu.py:14-24``) sees ``(tokens, hidden) @ (hidden, out)``;
``rms_norm_gpu`` (``:27-31``) reduces over the last dim only; ``swiglu_ffn_gpu``
(``:185-194``) is elementwise plus linears; the embedding gather
(``model_gpu.py:64``) indexes an id vector of any length. RoPE is *already*
batch-ready, because ``positions`` is a passed tensor indexed as ``cos[positions]``
(``components_gpu.py:132``) — feed it the right vector and it simply works.

A padded ``(batch, seq, hidden)`` layout would instead make every operation
batch-aware, waste compute proportional to length skew, and require threading a
padding mask everywhere. With skewed length distributions — which any honest
serving benchmark uses — that waste is large.

So the only genuinely batch-sensitive parts are attention, the cache write, the
causal mask, and the final "last token of each sequence" gather. Batched decode
and paged KV are not two projects; they are one change to one surface.

NOTHING IN THIS MODULE IMPORTS TORCH AT RUNTIME.
``import engine`` must stay usable on a CPU-only machine (see ``engine/__init__``),
so tensor types are referenced only under ``TYPE_CHECKING``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from torch import Tensor


__all__ = ["BatchMeta", "AttentionBackend"]


@dataclass(frozen=True)
class BatchMeta:
    """
    Everything a backend needs to know about one forward pass over a ragged batch.

    Frozen because it describes a single step and must not be mutated halfway
    through a 16-layer loop: every layer in one forward pass sees the same batch.

    All tensors live on the model's device unless noted. ``tokens`` below always
    means ``sum(query_lens)`` — the length of the packed token axis.

    FIELD-BY-FIELD, AND WHY EACH IS NEEDED
    --------------------------------------
    query_lens
        ``(n_seqs,) int32``. New tokens contributed by each sequence *this step*.
        A pure decode batch is all ones. A chunked prefill contributes many.
        Mixing the two in one batch is the point of continuous batching.

    cu_query_lens
        ``(n_seqs + 1,) int32``. Exclusive prefix sum of ``query_lens``, so
        sequence ``i`` owns the token slice ``[cu_query_lens[i], cu_query_lens[i+1])``.
        Standard varlen convention; attention kernels consume it directly rather
        than re-deriving segment boundaries.

    kv_lens
        ``(n_seqs,) int32``. Total KV length per sequence AFTER this step's append.
        This is what makes causality expressible without a materialized mask: a
        query at position ``p`` may attend to keys ``[0, p]``, and the backend
        derives that from ``kv_lens`` and ``query_lens``.

    positions
        ``(tokens,) int64``. RoPE position of every token, supplied BY THE CALLER.

        This is the field that fixes a latent bug. ``model_gpu.py`` computes
        ``torch.arange(seq)`` for prefill, starting at 0 regardless of how much
        history the cache already holds. Harmless today because prefill runs
        exactly once per request (``engine/scheduler.py:33``); wrong the moment a
        prompt is prefilled in chunks, where the second chunk would receive
        positions ``0..n`` instead of ``pos..pos+n``.

        The fix is not "add an offset inside the model". It is "stop computing
        positions inside the model at all" — the scheduler knows where each token
        sits in its sequence; the model should not have to guess.

    last_token_ix
        ``(n_seqs,) int32``. Index into the token axis of each sequence's final
        token — where next-token logits are needed. Replaces the ``x[-1:]`` slice
        (``model_gpu.py:88``): in a ragged batch "the last token" is a gather, not
        a slice, and for a mid-prefill chunk there is no logit wanted at all.

    PAGED KV ADDRESSING — CSR, NOT A PADDED MATRIX
    ----------------------------------------------
    kv_indptr, kv_indices, kv_last_page_len
        Compressed-sparse-row description of which physical pages hold each
        sequence's KV:

            sequence i owns pages  kv_indices[kv_indptr[i] : kv_indptr[i+1]]

        ``kv_indptr`` is ``(n_seqs + 1,) int32``; ``kv_indices`` is
        ``(total_pages,) int32``; ``kv_last_page_len`` is ``(n_seqs,) int32``.

        A padded ``(n_seqs, max_pages)`` block-table matrix is vLLM's convention.
        FlashInfer — the intended fast path — uses this CSR triple instead
        (verified against ``flashinfer-python==0.6.16``: ``flashinfer/page.py``
        ``append_paged_kv_cache(..., kv_indices, kv_indptr, kv_last_page_len)``).
        CSR is also strictly better here: no padding, no sentinel value, and it is
        produced from the allocator's natural per-sequence ``list[int]`` by a
        concatenation plus a prefix sum.

        INVARIANT WORTH MEMORISING: ``1 <= kv_last_page_len <= page_size``.
        A sequence whose length is an exact multiple of ``page_size`` reports
        ``page_size``, NOT ``0``. Getting this wrong silently truncates or extends
        attention by a whole page — no crash, plausible text, wrong output.

    batch_indices
        ``(tokens,) int32``. Which sequence each new token belongs to. Redundant
        with ``cu_query_lens`` in principle, but FlashInfer's append API consumes
        it directly (``flashinfer/page.py`` ``append_paged_kv_cache(append_key,
        append_value, batch_indices, positions, ...)``), and materializing it once
        per step is cheaper than every backend re-deriving it per layer.

    slot_mapping
        ``(tokens,) int32``, optional. Flat physical slot for each new token::

            slot = page_id * page_size + offset_within_page

        A pure-PyTorch backend scatters with this in one indexed write and never
        needs to reason about pages. FlashInfer does not use it. Both fields are
        carried and each backend takes what it needs — the alternative is forcing
        one backend to reconstruct the other's addressing on every layer.

    page_size
        Tokens per page. Needed to validate ``kv_last_page_len`` and to convert
        between the two addressing forms.

    is_prefill
        True when any sequence contributes more than one token. Backends may pick
        a different kernel for prefill-shaped versus decode-shaped work; it is a
        hint, never a correctness input.
    """

    query_lens: "Tensor"
    cu_query_lens: "Tensor"
    kv_lens: "Tensor"
    positions: "Tensor"
    last_token_ix: "Tensor"

    # Paged addressing (CSR)
    kv_indptr: "Tensor"
    kv_indices: "Tensor"
    kv_last_page_len: "Tensor"

    # Write addressing
    batch_indices: "Tensor"
    slot_mapping: "Tensor | None" = None

    page_size: int = 16
    is_prefill: bool = False

    @property
    def n_seqs(self) -> int:
        return int(self.query_lens.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.positions.shape[0])

    def validate(self) -> None:
        """
        Assert every structural invariant. Cheap relative to a forward pass, and
        each check corresponds to a failure mode that is otherwise SILENT —
        wrong attention, fluent output, no exception, no metric change.

        Call this in tests and in debug runs. It is not on the hot path by
        default, but it is deliberately cheap enough that it could be.
        """
        n = self.n_seqs
        t = self.n_tokens

        assert self.query_lens.shape == (n,), f"query_lens {tuple(self.query_lens.shape)} != ({n},)"
        assert self.kv_lens.shape == (n,), f"kv_lens {tuple(self.kv_lens.shape)} != ({n},)"
        assert self.last_token_ix.shape == (n,), "last_token_ix must be one index per sequence"
        assert self.cu_query_lens.shape == (n + 1,), "cu_query_lens must have n_seqs + 1 entries"
        assert self.batch_indices.shape == (t,), "batch_indices must be one entry per token"

        # cu_query_lens must be the exclusive prefix sum of query_lens.
        assert int(self.cu_query_lens[0]) == 0, "cu_query_lens must start at 0"
        assert int(self.cu_query_lens[-1]) == t, (
            f"cu_query_lens[-1]={int(self.cu_query_lens[-1])} must equal n_tokens={t}"
        )

        # CSR shape agreement.
        assert self.kv_indptr.shape == (n + 1,), "kv_indptr must have n_seqs + 1 entries"
        assert int(self.kv_indptr[0]) == 0, "kv_indptr must start at 0"
        assert int(self.kv_indptr[-1]) == int(self.kv_indices.shape[0]), (
            "kv_indptr[-1] must equal len(kv_indices)"
        )
        assert self.kv_last_page_len.shape == (n,), "kv_last_page_len must be one entry per sequence"

        # THE off-by-one. An exact multiple of page_size reports page_size, not 0.
        lo = int(self.kv_last_page_len.min())
        hi = int(self.kv_last_page_len.max())
        assert 1 <= lo and hi <= self.page_size, (
            f"kv_last_page_len must lie in [1, {self.page_size}], got [{lo}, {hi}]. "
            "A sequence whose length is an exact multiple of page_size reports "
            "page_size, NOT 0."
        )

        # Pages allocated must actually cover the KV each sequence claims to have.
        for i in range(n):
            pages = int(self.kv_indptr[i + 1]) - int(self.kv_indptr[i])
            kv_len = int(self.kv_lens[i])
            expected = (kv_len + self.page_size - 1) // self.page_size
            assert pages == expected, (
                f"seq {i}: {pages} pages allocated but kv_len={kv_len} needs {expected}"
            )
            last = int(self.kv_last_page_len[i])
            tail = kv_len - (pages - 1) * self.page_size
            assert last == tail, (
                f"seq {i}: kv_last_page_len={last} but kv_len={kv_len} over {pages} "
                f"pages implies {tail}"
            )

        if self.slot_mapping is not None:
            assert self.slot_mapping.shape == (t,), "slot_mapping must be one slot per token"


@runtime_checkable
class AttentionBackend(Protocol):
    """
    Owns KV storage and the attention computation for one model instance.

    A backend is stateful: it holds the physical KV pool for every layer. The
    engine holds no cache state at all on this path and never touches ``.k``,
    ``.v`` or ``.pos``.

    Two methods, called per layer, in this order, by ``gqa_attention_gpu``.
    Splitting them is not cosmetic: the append must complete before the attend so
    that a decoding token attends to *itself*, and a serving layer may need to
    append without attending (speculative rollback, prefix warming).

    ``runtime_checkable`` so a serving layer can ``isinstance``-check a plugin at
    startup rather than discovering a missing method mid-benchmark. Note that
    this only verifies method *presence*, never signatures.
    """

    def append_kv(
        self,
        layer_idx: int,
        k: "Tensor",
        v: "Tensor",
        meta: BatchMeta,
    ) -> None:
        """
        Write this step's K/V into the physical pool.

        ``k`` and ``v`` are ``(tokens, n_kv_heads, head_dim)``, RoPE already
        applied to ``k`` — matching what the engine computes at
        ``components_gpu.py:128-135``. Token ``j`` belongs to sequence
        ``meta.batch_indices[j]`` at position ``meta.positions[j]``.

        Must be idempotent for a given ``(layer_idx, meta)``: a scheduler that
        retries a step after a recoverable failure must not double-append.
        """
        ...

    def attend(
        self,
        q: "Tensor",
        layer_idx: int,
        scale: float,
        meta: BatchMeta,
    ) -> "Tensor":
        """
        Attention over each sequence's full KV history, including what
        ``append_kv`` just wrote.

        ``q`` is ``(tokens, n_heads, head_dim)`` with RoPE applied; the return is
        the same shape. GQA mapping (query head ``h`` reads KV head
        ``h // (n_heads // n_kv_heads)``) is the backend's responsibility, as is
        causality — the engine builds no mask on this path.

        Causality is expressible from ``meta`` alone: sequence ``i`` contributes
        ``query_lens[i]`` queries whose true positions end at ``kv_lens[i] - 1``,
        so query ``j`` within that sequence attends to keys
        ``[0, kv_lens[i] - query_lens[i] + j]``.
        """
        ...
