"""Decode latency / throughput / KV-bytes vs context length.

The headline efficiency experiment. Sparse decode only pays off once the KV
cache is long enough to amortise the selection overhead, so we sweep context
length L and report where AdaKV overtakes dense and how it compares to Quest and
StreamingLLM at a *matched* KV budget.

Why this runs where the quality eval OOMs: decode attention for a single query
token is O(L) memory (scores are [heads, L], not [L, L]), and we synthesise the
KV cache directly instead of running a dense prefill. So 16K / 32K / 64K all fit
on a small GPU (e.g. T4), and no dense L x L matrix is ever materialised.

Methods (all read the SAME synthesized cache, matched avg_budget blocks/head):
    dense       full attention over all L tokens (grouped GQA einsum)
    streaming   sink + most-recent window blocks (no scoring)          [StreamingLLM]
    quest       minmax estimator + fixed top-k                          [Quest]
    adakv       centroid estimator + nucleus adaptive budget            [ours]

Reports per (L, method): median ms/token, tokens/s, speedup vs dense, mean
blocks/head, KV bytes read/token, and the estimator's summary memory (centroid
is 1xD/block, Quest's min/max is 2xD/block -> AdaKV's summary is half the size).

Run (Colab T4):
    python benchmarks/efficiency/bench_decode_latency.py \
        --ctx 4096 8192 16384 32768 --budget-blocks 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from adakv.attention import AdaKVCache, plan_selection
from adakv.kernels.block_sparse_decode import HAS_TRITON, block_sparse_decode
from adakv.runtime import _torch_masked_decode


def dense_decode(q, k, v):
    """Grouped-GQA dense decode attention. q:[Hq,D], k,v:[Hkv,L,D] -> [Hq,D].

    Uses the [Hkv,g,L] score layout so the [Hq,L] tensor is never materialised
    per query head redundantly -- fits at 32K+ on a small GPU.
    """
    Hkv, L, D = k.shape
    Hq = q.shape[0]
    g = Hq // Hkv
    qg = q.view(Hkv, g, D).float()
    a = torch.einsum("hgd,hsd->hgs", qg, k.float()) / (D ** 0.5)
    w = torch.softmax(a, dim=-1)
    o = torch.einsum("hgs,hsd->hgd", w, v.float())
    return o.reshape(Hq, D).to(q.dtype)


def streaming_table(nb, budget, sink, Hq, device):
    """StreamingLLM selection: sink blocks + most-recent window. No scoring."""
    w = max(budget - sink, 1)
    idx = torch.cat([torch.arange(0, sink, device=device),
                     torch.arange(nb - w, nb, device=device)])[:budget]
    bt = idx.to(torch.int32).unsqueeze(0).expand(Hq, -1).contiguous()
    sl = torch.full((Hq,), idx.numel(), dtype=torch.int32, device=device)
    return bt, sl


def sparse_attn(qh, cache, bt, sl):
    if HAS_TRITON and qh.is_cuda:
        return block_sparse_decode(qh, cache.k, cache.v, bt, sl, cache.block_size)
    return _torch_masked_decode(qh, cache.k, cache.v, bt, sl,
                                cache.block_size, cache.centroid.shape[1])


def timeit(fn, warmup, iters, device):
    """Median ms per call."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iters)]
        for s, e in ev:
            s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) for s, e in ev]
    else:
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def summary_mb(t):
    return t.element_size() * t.nelement() / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--budget-blocks", type=int, default=64, help="avg KV blocks/head (matched)")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--heads-q", type=int, default=12)
    ap.add_argument("--heads-kv", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--n-local", type=int, default=2)
    ap.add_argument("--c-min", type=int, default=1)
    ap.add_argument("--nucleus-p", type=float, default=0.9)
    ap.add_argument("--k-max", type=int, default=100000)
    ap.add_argument("--methods", nargs="+", default=["dense", "streaming", "quest", "adakv"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    Hq, Hkv, D, bs = args.heads_q, args.heads_kv, args.head_dim, args.block_size
    B = args.budget_blocks
    kernel = "triton" if (HAS_TRITON and device == "cuda") else "torch-fallback"
    print(f"device {device} | dtype {dtype} | attn backend {kernel}")
    print(f"Hq {Hq} Hkv {Hkv} D {D} | block_size {bs} | budget {B} blocks/head "
          f"(= {B*bs} tokens) | sink {args.n_sink} local {args.n_local}\n")

    hdr = (f"{'ctx':>6} {'method':<10} {'ms/tok':>8} {'tok/s':>8} {'vs_dense':>9} "
           f"{'blk/hd':>7} {'KV_MB/tok':>10} {'summ_MB':>8}")
    print(hdr); print("-" * len(hdr))

    for L in args.ctx:
        k = torch.randn(Hkv, L, D, device=device, dtype=dtype)
        v = torch.randn(Hkv, L, D, device=device, dtype=dtype)
        q = torch.randn(Hq, D, device=device, dtype=dtype)
        cache_c = AdaKVCache(bs, "centroid").append_prefill(k, v)
        cache_m = AdaKVCache(bs, "minmax").append_prefill(k, v)
        nb = cache_c.centroid.shape[1]
        bt_s, sl_s = streaming_table(nb, B, args.n_sink, Hq, device)
        common = dict(k_min=1, k_max=args.k_max, n_sink_blocks=args.n_sink,
                      n_local_blocks=args.n_local, c_min=args.c_min, nucleus_p=args.nucleus_p)

        def run_dense():
            return dense_decode(q, k, v)

        def run_streaming():
            return sparse_attn(q, cache_c, bt_s, sl_s)

        def run_quest():
            bt, sl = plan_selection(q, cache_m, B, estimator="minmax",
                                    budget_policy="fixed", **common)
            return sparse_attn(q, cache_m, bt, sl)

        def run_adakv():
            bt, sl = plan_selection(q, cache_c, B, estimator="centroid",
                                    budget_policy="adaptive_nucleus", **common)
            return sparse_attn(q, cache_c, bt, sl)

        runners = {"dense": run_dense, "streaming": run_streaming,
                   "quest": run_quest, "adakv": run_adakv}
        summ = {"dense": 0.0, "streaming": summary_mb(cache_c.centroid),
                "quest": summary_mb(cache_m.kmin) + summary_mb(cache_m.kmax),
                "adakv": summary_mb(cache_c.centroid)}
        # realized blocks/head (dense reads all)
        blk = {"dense": float(nb), "streaming": float(int(sl_s[0]))}
        for name in ("quest", "adakv"):
            est = "minmax" if name == "quest" else "centroid"
            pol = "fixed" if name == "quest" else "adaptive_nucleus"
            cache = cache_m if name == "quest" else cache_c
            _, sl = plan_selection(q, cache, B, estimator=est, budget_policy=pol, **common)
            blk[name] = float(sl.float().mean().item())

        dense_ms = None
        for name in args.methods:
            ms = timeit(runners[name], args.warmup, args.iters, device)
            if name == "dense":
                dense_ms = ms
            blocks = blk[name]
            kv_mb = Hkv * blocks * bs * D * 2 * k.element_size() / 1e6   # k+v bytes read
            spd = (dense_ms / ms) if dense_ms else float("nan")
            print(f"{L:>6} {name:<10} {ms:>8.3f} {1000/ms:>8.1f} {spd:>8.2f}x "
                  f"{blocks:>7.1f} {kv_mb:>10.2f} {summ[name]:>8.2f}")
        print()
        del k, v, cache_c, cache_m
        if device == "cuda":
            torch.cuda.empty_cache()

    print("Read: at long ctx, dense ms/tok grows with L (reads all KV) while "
          "sparse stays flat (reads ~budget). adakv vs quest at equal budget: same "
          "attention cost, but centroid scoring is cheaper and its summary is half "
          "the size (summ_MB column).")


if __name__ == "__main__":
    main()
