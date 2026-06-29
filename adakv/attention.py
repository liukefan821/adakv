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


class AdaKVCache:
    """Full KV cache plus precomputed block summaries (no permanent eviction)."""

    def __init__(self, block_size: int, estimator: str = "centroid"):
        self.block_size = block_size
        self.estimator = estimator
        self.k = None  # [n_kv_heads, S, D]
        self.v = None
        self.centroid = None  # [n_kv_heads, n_blocks, D]

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
):
    """Score blocks, allocate an adaptive per-head budget, select blocks.

    Returns (block_table [int32, Hq x max_sel], sel_lens [int32, Hq]) ready for
    the kernel. sel_lens >= n_sink_blocks + n_local_blocks >= 1.
    """
    Hq, D = q.shape
    Hkv, nb, _ = cache.centroid.shape
    group = Hq // Hkv

    cent = cache.centroid.repeat_interleave(group, dim=0)          # [Hq, nb, D]
    scores = torch.einsum("hd,hbd->hb", q.float(), cent.float())   # [Hq, nb]

    # adaptive budget from block-score concentration (normalised entropy)
    p = torch.softmax(scores / max(temperature, 1e-6), dim=-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(max(nb, 2))
    raw = k_min + ent * (min(k_max, nb) - k_min)
    raw = raw * (avg_budget / raw.mean().clamp_min(1e-6))
    kph = raw.round().clamp(k_min, min(k_max, nb)).long()          # [Hq]

    # force sink + local, fill remaining budget with top-scoring blocks
    sink, local = min(n_sink_blocks, nb), min(n_local_blocks, nb)
    biased = scores.clone()
    if sink:
        biased[:, :sink] = float("inf")
    if local:
        biased[:, nb - local :] = float("inf")
    order = biased.argsort(dim=-1, descending=True)                # [Hq, nb] permutation

    max_sel = int(kph.clamp_min(sink + local).max().item())
    block_table = torch.zeros(Hq, max_sel, dtype=torch.int32, device=q.device)
    sel_lens = torch.zeros(Hq, dtype=torch.int32, device=q.device)
    for h in range(Hq):
        kh = max(int(kph[h]), sink + local)
        block_table[h, :kh] = order[h, :kh].to(torch.int32)
        sel_lens[h] = kh
    # NOTE: this host loop is the obvious next thing to vectorise / fuse.
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
