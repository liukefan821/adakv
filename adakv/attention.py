"""Runtime attention module (torch) wiring selection -> kernel.

This is the drop-in unit that ``patch.py`` swaps into a HuggingFace model. The
cheap parts (block scoring, budget, selection) run as small torch ops on the
host every decode step; the heavy part (gather + attention over selected
blocks) is the Triton kernel in ``kernels.block_sparse_decode``.

Selection is factored into ``plan_selection`` so it can be unit-tested and
reused by benchmarks independently of the kernel.

STUB: the prefill path / full-prefill attention are TODOs -- prefill is dense
and only the *decode* path uses sparse selection, since that is where a long KV
cache makes attention memory-bound.
"""
from __future__ import annotations

import math

import torch

from .kernels.block_sparse_decode import block_sparse_decode


# --- optional realized-budget instrumentation -------------------------------
# When tracing is on, every plan_selection call appends its mean blocks/head.
# Used by benchmarks to prove that two methods ran at an equal *realized* KV
# budget on the actual model trace (not just an equal target).
_BUDGET_TRACE = None


def reset_budget_trace():
    """Start (or restart) recording realized mean blocks/head per decode step."""
    global _BUDGET_TRACE
    _BUDGET_TRACE = []


def get_budget_trace():
    """Return the list of per-step mean blocks/head recorded since reset."""
    return list(_BUDGET_TRACE) if _BUDGET_TRACE is not None else []


def stop_budget_trace():
    """Disable recording (zero overhead in plan_selection)."""
    global _BUDGET_TRACE
    _BUDGET_TRACE = None


# --- optional selection-recall instrumentation ------------------------------
# Measures whether the block(s) holding a planted fact survive selection. For
# each decode step, a target block is "covered" if at least one head selects it.
# Far lower variance than end-to-end answer accuracy, so it isolates *selection*
# quality from generation noise. The harness sets the target block ids for the
# current sequence, generates, then reads the recall fraction.
_RECALL_TARGET = None   # set[int] of target block indices, or None
_RECALL_HITS = 0        # covered (step, target) pairs
_RECALL_TOTAL = 0       # total (step, target) pairs


def set_recall_target(block_ids):
    """Begin recording coverage of these block indices; resets counters."""
    global _RECALL_TARGET, _RECALL_HITS, _RECALL_TOTAL
    _RECALL_TARGET = set(int(b) for b in block_ids)
    _RECALL_HITS = 0
    _RECALL_TOTAL = 0


def get_recall():
    """Fraction of (decode-step, target-block) pairs covered by >=1 head."""
    return (_RECALL_HITS / _RECALL_TOTAL) if _RECALL_TOTAL else float("nan")


def clear_recall_target():
    global _RECALL_TARGET
    _RECALL_TARGET = None


def _record_recall(block_table, sel_lens, nb):
    """Tally, for each target block, whether any head selected it this step."""
    global _RECALL_HITS, _RECALL_TOTAL
    import torch as _t
    Hq = block_table.shape[0]
    sel = _t.zeros(nb, dtype=_t.bool, device=block_table.device)
    for h in range(Hq):
        sel[block_table[h, : int(sel_lens[h])].long()] = True
    for b in _RECALL_TARGET:
        if 0 <= b < nb:
            _RECALL_TOTAL += 1
            _RECALL_HITS += int(bool(sel[b].item()))


class AdaKVCache:
    """Full KV cache plus precomputed block summaries (no permanent eviction)."""

    def __init__(self, block_size: int, estimator: str = "centroid"):
        self.block_size = block_size
        self.estimator = estimator
        self.k = None  # [n_kv_heads, S, D]
        self.v = None
        self.centroid = None  # [n_kv_heads, n_blocks, D]
        # Quest-style per-dimension block bounds; built only when estimator=="minmax".
        self.kmin = None  # [n_kv_heads, n_blocks, D]
        self.kmax = None  # [n_kv_heads, n_blocks, D]

    def append_prefill(self, k, v):
        """Store prefill KV and (re)compute block centroids. k,v: [Hkv,S,D]."""
        self.k, self.v = k, v
        Hkv, S, D = k.shape
        nb = (S + self.block_size - 1) // self.block_size
        pad = nb * self.block_size - S
        kk = torch.nn.functional.pad(k, (0, 0, 0, pad)) if pad else k
        # mean over *real* tokens per block (last block may be partial)
        counts = torch.full((nb,), self.block_size, device=k.device, dtype=kk.dtype)
        if pad:
            counts[-1] = self.block_size - pad
        self.centroid = kk.view(Hkv, nb, self.block_size, D).sum(dim=2) / counts.view(1, nb, 1)
        # Quest baseline estimator: per-block per-dim min/max over *real* tokens.
        # Pad with +inf for the min reduction and -inf for the max reduction so the
        # partial trailing block's padding never enters the statistic.
        if self.estimator == "minmax":
            kmax_pad = torch.nn.functional.pad(k, (0, 0, 0, pad), value=float("-inf")) if pad else k
            kmin_pad = torch.nn.functional.pad(k, (0, 0, 0, pad), value=float("inf")) if pad else k
            self.kmax = kmax_pad.view(Hkv, nb, self.block_size, D).amax(dim=2)
            self.kmin = kmin_pad.view(Hkv, nb, self.block_size, D).amin(dim=2)
        return self

    def append_decode(self, k_t, v_t):
        """Append one decoded token's KV. k_t,v_t: [Hkv,1,D]."""
        # TODO: incremental block-summary update for the trailing (partial) block.
        self.k = torch.cat([self.k, k_t], dim=1)
        self.v = torch.cat([self.v, v_t], dim=1)
        return self


def plan_selection(
    q,                      # [n_q_heads, D]
    cache: AdaKVCache,
    avg_budget: int,
    k_min: int = 2,
    k_max: int = 64,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    temperature: float = 1.0,
    estimator: str = "centroid",
    budget_policy: str = "adaptive",
    c_min: int = 1,
    nucleus_p: float = 0.9,
):
    """Score blocks, allocate a per-head budget, select blocks.

    The (estimator, budget_policy) pair selects one cell of the comparison grid:
      - ("centroid", "adaptive")         -> AdaKV (entropy budget)
      - ("centroid", "adaptive_nucleus") -> AdaKV (nucleus budget)
      - ("minmax",   "fixed")            -> Quest baseline (in-framework, same kernel)
      - ("centroid", "fixed")            -> estimator-only ablation
      - ("minmax",   "adaptive")         -> budget-only ablation

    All budget policies pin the *mean* budget to ``avg_budget`` and every head is
    guaranteed at least ``c_min`` content blocks beyond the sink/local stabilisers
    (the effective floor is ``sink + local + c_min``), so no head can be silently
    reduced to a pure sliding window. Only the scoring and the per-head count
    differ across cells; the gather and the kernel are identical, so an
    equal-budget quality comparison isolates the two algorithmic contributions.

    Returns (block_table [int32, Hq x max_sel], sel_lens [int32, Hq]) ready for
    the kernel.
    """
    Hq, D = q.shape
    Hkv, nb, _ = cache.centroid.shape
    group = Hq // Hkv

    # --- block scores -------------------------------------------------------
    if estimator == "centroid":
        cent = cache.centroid.repeat_interleave(group, dim=0)          # [Hq, nb, D]
        scores = torch.einsum("hd,hbd->hb", q.float(), cent.float())   # [Hq, nb]
    elif estimator == "minmax":
        if cache.kmin is None or cache.kmax is None:
            raise ValueError("minmax estimator needs AdaKVCache(estimator='minmax')")
        kmn = cache.kmin.repeat_interleave(group, dim=0).float()       # [Hq, nb, D]
        kmx = cache.kmax.repeat_interleave(group, dim=0).float()
        qf = q.float()
        qpos = qf.clamp_min(0.0)[:, None, :]
        qneg = qf.clamp_max(0.0)[:, None, :]
        scores = (qpos * kmx + qneg * kmn).sum(-1)                     # Quest upper bound
    else:
        raise ValueError(f"unknown estimator: {estimator!r}")

    kmax_eff = min(k_max, nb)
    sink, local = min(n_sink_blocks, nb), min(n_local_blocks, nb)
    floor = min(sink + local + c_min, nb)
    lo = min(max(floor, k_min, 1), nb)

    # --- per-head budget (mean pinned to avg_budget for every policy) --------
    # Every head is guaranteed `lo` blocks; the surplus (avg_budget-lo)*Hq is
    # distributed by a per-head weight, so the realized mean equals avg_budget
    # exactly (no one-sided floor clamp -> no budget overspend).
    target = float(min(max(avg_budget, lo), kmax_eff))
    if budget_policy == "fixed":
        kph = torch.full((Hq,), int(round(target)), dtype=torch.long, device=q.device)
    elif budget_policy in ("adaptive", "adaptive_nucleus"):
        p = torch.softmax(scores / max(temperature, 1e-6), dim=-1)
        if budget_policy == "adaptive":
            w = -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(max(nb, 2))   # entropy
            w = w + 1e-6
        else:
            ps, _ = torch.sort(p, dim=-1, descending=True)
            w = ((ps.cumsum(dim=-1) < nucleus_p).sum(dim=-1).float() + 1.0)      # nucleus
        surplus_total = max(target - lo, 0.0) * Hq
        extra = surplus_total * w / w.sum().clamp_min(1e-9)
        kph = (lo + extra).round().clamp(lo, kmax_eff).long()
    else:
        raise ValueError(f"unknown budget_policy: {budget_policy!r}")

    # --- force sink + local, fill remaining budget with top-scoring blocks ---
    biased = scores.clone()
    if sink:
        biased[:, :sink] = float("inf")
    if local:
        biased[:, nb - local :] = float("inf")
    order = biased.argsort(dim=-1, descending=True)                    # [Hq, nb] permutation

    max_sel = min(int(kph.clamp_min(floor).max().item()), nb)
    block_table = torch.zeros(Hq, max_sel, dtype=torch.int32, device=q.device)
    sel_lens = torch.zeros(Hq, dtype=torch.int32, device=q.device)
    for h in range(Hq):
        kh = min(max(int(kph[h]), floor), nb)
        block_table[h, :kh] = order[h, :kh].to(torch.int32)
        sel_lens[h] = kh
    # NOTE: this host loop is the obvious next thing to vectorise / fuse.

    if _BUDGET_TRACE is not None:
        _BUDGET_TRACE.append(float(sel_lens.float().mean().item()))
    if _RECALL_TARGET is not None:
        _record_recall(block_table, sel_lens, nb)
    return block_table, sel_lens


def adakv_decode_attention(
    q,                      # [n_q_heads, D]
    cache: AdaKVCache,
    avg_budget: int,
    k_min: int = 2,
    k_max: int = 64,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    temperature: float = 1.0,
):
    """One decode step of AdaKV attention: plan selection, then run the kernel."""
    block_table, sel_lens = plan_selection(
        q, cache, avg_budget, k_min, k_max, n_sink_blocks, n_local_blocks, temperature
    )
    D = q.shape[-1]
    return block_sparse_decode(
        q, cache.k, cache.v, block_table, sel_lens, cache.block_size,
        sm_scale=1.0 / (D ** 0.5),
    )
