"""Block selection.

Given per-head scores and a per-head budget, decide which KV blocks each head
attends to. We always force the first ``n_sink_blocks`` (attention sinks) and
the last ``n_local_blocks`` (sliding window) to be selected -- the StreamingLLM
observation that these stabilise long-context decoding -- and fill the remaining
budget with the highest-scoring blocks.

The reference uses a per-head Python loop for clarity. The Triton kernel does
the equivalent selection vectorised and fused with the gather; this file defines
the exact semantics it must reproduce.
"""
from __future__ import annotations

import numpy as np


def select_blocks(
    scores: np.ndarray,
    k_per_head: np.ndarray,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a boolean block mask and the per-head selected count.

    Args:
        scores: ``[n_heads, n_blocks]``.
        k_per_head: ``[n_heads]`` budget (from :func:`adakv.budget.allocate_budget`).
        n_sink_blocks: forced leading blocks.
        n_local_blocks: forced trailing blocks.

    Returns:
        ``mask``: ``[n_heads, n_blocks]`` bool, ``counts``: ``[n_heads]`` int.
    """
    H, B = scores.shape
    sink = min(n_sink_blocks, B)
    local = min(n_local_blocks, B)

    biased = scores.astype(np.float64, copy=True)
    if sink:
        biased[:, :sink] = np.inf
    if local:
        biased[:, B - local:] = np.inf

    order = np.argsort(-biased, axis=1)  # descending
    mask = np.zeros((H, B), dtype=bool)
    floor = sink + local
    for h in range(H):
        kh = max(int(k_per_head[h]), floor)
        mask[h, order[h, :kh]] = True
        if sink:
            mask[h, :sink] = True
        if local:
            mask[h, B - local:] = True
    return mask, mask.sum(axis=1)
