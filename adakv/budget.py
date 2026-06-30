"""Adaptive per-head budget allocation.

This is the part of AdaKV that differs most from prior query-aware sparse
attention (Quest uses a *fixed* top-k per head/layer). Intuition: a "retrieval"
head whose block-score distribution is flat needs many blocks to avoid dropping
the relevant one; a "local" head whose scores are sharply peaked needs only a
few. We turn a *global* KV budget into a *per-head* budget driven by a cheap,
training-free concentration signal (normalised entropy of the block-score
softmax), then rescale so the mean budget matches the target.

Everything here is O(n_heads * n_blocks) and runs on the host every decode step;
it is negligible next to the attention itself.
"""
from __future__ import annotations

import numpy as np


def allocate_budget(
    scores: np.ndarray,
    avg_budget: int,
    k_min: int,
    k_max: int,
    temperature: float = 1.0,
    policy: str = "adaptive",
    nucleus_p: float = 0.9,
) -> np.ndarray:
    """Allocate a per-head *total* block budget from a global average.

    All policies pin the mean budget to ``avg_budget`` (so two policies can be
    compared at an equal KV budget); they differ only in how budget is *shaped*
    across heads. ``selector.select_blocks`` then enforces the sink+local+c_min
    floor, so a per-head value below the floor is raised (never silently zeroed).

    Args:
        scores: ``[n_heads, n_blocks]`` block importance scores.
        avg_budget: target *mean* number of blocks per head.
        k_min, k_max: per-head clamps (k_max is also capped at n_blocks by caller).
        temperature: softmax temperature for the concentration estimate.
        policy:
            - ``"fixed"``     : same budget for every head (Quest-style top-k).
            - ``"adaptive"``  : entropy of the block-score softmax; flat heads get
              more, peaked heads fewer.
            - ``"adaptive_nucleus"`` : number of blocks needed to capture
              ``nucleus_p`` of the attention mass; sizes budget to each head's
              concentration directly.
        nucleus_p: cumulative-mass target for the nucleus policy.

    Returns:
        ``[n_heads]`` int64 total budget per head, clamped to ``[k_min, k_max]``
        and rescaled so ``mean(k) ~= avg_budget``.
    """
    H, B = scores.shape
    if B <= 1:
        return np.full((H,), min(max(k_min, 1), B), dtype=np.int64)

    if policy == "fixed":
        return np.full((H,), int(np.clip(avg_budget, k_min, k_max)), dtype=np.int64)

    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)

    if policy == "adaptive":
        # Normalised entropy in [0, 1]: 0 = fully peaked, 1 = uniform.
        ent = -(p * np.log(np.clip(p, 1e-9, None))).sum(axis=-1) / np.log(B)
        raw = k_min + ent * (k_max - k_min)
    elif policy == "adaptive_nucleus":
        ps = np.sort(p, axis=-1)[:, ::-1]          # descending per head
        cum = np.cumsum(ps, axis=-1)
        # blocks needed to reach nucleus_p (>=1)
        raw = (cum < nucleus_p).sum(axis=-1).astype(np.float64) + 1.0
    else:
        raise ValueError(f"unknown budget policy: {policy!r}")

    raw = raw * (avg_budget / max(float(raw.mean()), 1e-6))
    return np.clip(np.round(raw), k_min, k_max).astype(np.int64)
