"""Reference (oracle) attention in pure NumPy.

This ties the pieces together into a single decode step and is the numerical
ground truth for the Triton kernel. It deliberately computes the *full* score
row and then masks the non-selected positions, because that makes correctness
obvious. The Triton kernel must produce the same output while only ever reading
the *selected* KV blocks (that is the whole point -- O(budget) memory traffic
instead of O(seq_len)).

Layout convention everywhere: ``[n_heads, seq_len, head_dim]``. KV is expanded
to query-head count via :func:`expand_gqa` before scoring/attention, so GQA/MQA
collapses to the dense-head case here.
"""
from __future__ import annotations

import numpy as np

from .budget import allocate_budget
from .estimator import block_scores, build_block_summaries
from .selector import select_blocks


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expand_gqa(x: np.ndarray, n_q_heads: int) -> np.ndarray:
    """Expand a KV-head-indexed array to query-head count by group repeat."""
    h_kv = x.shape[0]
    assert n_q_heads % h_kv == 0, "n_q_heads must be a multiple of n_kv_heads"
    return np.repeat(x, n_q_heads // h_kv, axis=0)


def dense_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Exact single-query attention. q:[H,D]  K,V:[H,S,D] -> [H,D]."""
    D = q.shape[-1]
    a = np.einsum("hd,hsd->hs", q, K) / np.sqrt(D)
    w = _softmax(a, axis=-1)
    return np.einsum("hs,hsd->hd", w, V)


def sparse_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 16,
    avg_budget: int = 8,
    k_min: int = 2,
    k_max: int = 64,
    estimator: str = "centroid",
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
    budget_policy: str = "adaptive",
    nucleus_p: float = 0.9,
    return_mask: bool = False,
):
    """AdaKV decode-step attention oracle.

    q:[H,D], K,V:[H,S,D] already expanded to query heads.

    Returns ``(out[H,D], counts[H])`` and optionally the block ``mask[H,n_blocks]``.
    """
    Hq, S, D = K.shape
    summ = build_block_summaries(K, block_size, estimator)
    scores = block_scores(q, summ)               # [H, n_blocks]
    n_blocks = summ["n_blocks"]

    k = allocate_budget(scores, avg_budget, min(k_min, n_blocks), min(k_max, n_blocks),
                        policy=budget_policy, nucleus_p=nucleus_p)
    mask, counts = select_blocks(scores, k, n_sink_blocks, n_local_blocks, c_min)

    tok = np.repeat(mask, block_size, axis=1)[:, :S]  # block mask -> token mask
    a = np.einsum("hd,hsd->hs", q, K) / np.sqrt(D)
    a = np.where(tok, a, -np.inf)
    w = _softmax(a, axis=-1)
    out = np.einsum("hs,hsd->hd", w, V)

    if return_mask:
        return out, counts, mask
    return out, counts
