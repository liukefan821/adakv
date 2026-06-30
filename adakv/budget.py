"""Adaptive per-head budget allocation (exact-mean, floor-aware).

This is the part of AdaKV that differs most from prior query-aware sparse
attention (Quest uses a *fixed* top-k per head/layer). The allocation is built so
that the realized mean budget equals the target *exactly*, which is what makes an
equal-budget comparison against Quest auditable:

    every head is guaranteed a floor of `floor` blocks (sink + local + c_min),
    and the remaining surplus  (avg_budget - floor) * n_heads  is distributed
    across heads by a training-free per-head weight.

Two adaptive weights are offered:
  - "adaptive"          : normalised entropy of the block-score softmax (flat
                          heads, which spread attention, get more surplus);
  - "adaptive_nucleus"  : number of blocks needed to capture `nucleus_p` of the
                          attention mass (a direct "how many blocks does this head
                          actually need" signal).

Because surplus is non-negative and sums to exactly (avg_budget - floor) * H, the
mean is pinned without the one-sided floor clamp that previously caused budget
overspend. Everything is O(n_heads * n_blocks) on the host per decode step.
"""
from __future__ import annotations

import numpy as np


def _weights(scores: np.ndarray, policy: str, temperature: float, nucleus_p: float) -> np.ndarray:
    """Per-head non-negative surplus weights."""
    B = scores.shape[1]
    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    if policy == "adaptive":
        ent = -(p * np.log(np.clip(p, 1e-9, None))).sum(axis=-1) / np.log(max(B, 2))
        return ent + 1e-6                       # entropy in [0,1], strictly positive
    if policy == "adaptive_nucleus":
        ps = np.sort(p, axis=-1)[:, ::-1]
        cum = np.cumsum(ps, axis=-1)
        return (cum < nucleus_p).sum(axis=-1).astype(np.float64) + 1.0
    raise ValueError(f"unknown budget policy: {policy!r}")


def allocate_budget(
    scores: np.ndarray,
    avg_budget: int,
    k_min: int,
    k_max: int,
    temperature: float = 1.0,
    policy: str = "adaptive",
    nucleus_p: float = 0.9,
    floor: int | None = None,
) -> np.ndarray:
    """Allocate a per-head *total* block budget with the mean pinned to avg_budget.

    Args:
        scores: ``[n_heads, n_blocks]`` block importance scores.
        avg_budget: target *mean* total blocks per head.
        k_min, k_max: per-head clamps (k_max also capped at n_blocks by caller).
        temperature: softmax temperature for the concentration estimate.
        policy: ``"fixed"`` | ``"adaptive"`` | ``"adaptive_nucleus"``.
        nucleus_p: cumulative-mass target for the nucleus policy.
        floor: per-head guaranteed minimum (sink + local + c_min). Defaults to
            ``k_min`` when not supplied. Surplus is distributed *above* this.

    Returns:
        ``[n_heads]`` int64 total budget per head; ``mean ~= avg_budget`` exactly
        (up to rounding), every head ``>= max(floor, k_min)``.
    """
    H, B = scores.shape
    if B <= 1:
        return np.full((H,), min(max(k_min, 1), B), dtype=np.int64)

    lo = min(max(int(k_min if floor is None else floor), int(k_min), 1), B)
    hi = min(int(k_max), B)
    target = float(np.clip(avg_budget, lo, hi))

    if policy == "fixed":
        return np.full((H,), int(round(target)), dtype=np.int64)

    w = _weights(scores, policy, temperature, nucleus_p)
    surplus_total = max(target - lo, 0.0) * H
    extra = surplus_total * w / max(float(w.sum()), 1e-9)
    k = np.clip(np.round(lo + extra), lo, hi).astype(np.int64)
    return k
