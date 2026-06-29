"""Block-level importance estimation.

This module is the *algorithmic spec / oracle* for how AdaKV scores KV blocks
given a decode query. It is written in pure NumPy so it can be unit-tested on
CPU and serve as the ground-truth the Triton kernel must match.

Two estimators are provided:

- ``centroid`` (default, our cheaper/sharper estimator): summarise each KV block
  by its mean key vector. Decode-time block score is a single dot product
  ``q . centroid[block]``. Cost: O(n_blocks * d) per query head, and the summary
  is O(n_blocks * d) memory vs O(seq_len * d) for the full cache.

- ``minmax`` (Quest baseline): store per-dimension min/max of the keys in a
  block and bound the *max* possible ``q . k`` inside the block. More memory
  (2x) and a looser ranking signal; included for ablation against Quest.

All tensors here are laid out as ``[n_heads, seq_len, head_dim]``. GQA/MQA is
handled by the caller (it expands KV-head summaries to query-head count before
scoring); see ``adakv.reference``.
"""
from __future__ import annotations

import numpy as np


def build_block_summaries(keys: np.ndarray, block_size: int, kind: str = "centroid") -> dict:
    """Precompute per-block summaries once, at the end of prefill.

    Args:
        keys: ``[n_heads, seq_len, head_dim]`` key cache.
        block_size: number of tokens per block (page). Must match the kernel.
        kind: ``"centroid"`` or ``"minmax"``.

    Returns:
        dict carrying ``kind``, ``n_blocks``, ``n_real`` (true seq_len) and the
        summary arrays. Partial trailing blocks are summarised over real tokens
        only (no zero padding leaking into the statistics).
    """
    H, S, D = keys.shape
    n_blocks = (S + block_size - 1) // block_size

    if kind == "centroid":
        c = np.zeros((H, n_blocks, D), dtype=np.float32)
        for b in range(n_blocks):
            s, e = b * block_size, min((b + 1) * block_size, S)
            c[:, b] = keys[:, s:e].mean(axis=1)
        return {"kind": "centroid", "n_blocks": n_blocks, "n_real": S, "centroid": c}

    if kind == "minmax":
        kmin = np.zeros((H, n_blocks, D), dtype=np.float32)
        kmax = np.zeros((H, n_blocks, D), dtype=np.float32)
        for b in range(n_blocks):
            s, e = b * block_size, min((b + 1) * block_size, S)
            kmin[:, b] = keys[:, s:e].min(axis=1)
            kmax[:, b] = keys[:, s:e].max(axis=1)
        return {"kind": "minmax", "n_blocks": n_blocks, "n_real": S, "kmin": kmin, "kmax": kmax}

    raise ValueError(f"unknown estimator kind: {kind!r}")


def block_scores(q: np.ndarray, summ: dict) -> np.ndarray:
    """Score every block for the current decode query.

    Args:
        q: ``[n_heads, head_dim]`` (already expanded to query-head count).
        summ: output of :func:`build_block_summaries`, also expanded to
            query-head count.

    Returns:
        ``[n_heads, n_blocks]`` importance scores (higher = more relevant).
    """
    if summ["kind"] == "centroid":
        return np.einsum("hd,hbd->hb", q, summ["centroid"])

    # Quest-style upper bound on max_{k in block} q . k.
    qpos = np.clip(q, 0.0, None)[:, None, :]
    qneg = np.clip(q, None, 0.0)[:, None, :]
    return (qpos * summ["kmax"] + qneg * summ["kmin"]).sum(-1)
