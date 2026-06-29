"""CPU integration tests for the AdaKV decode runtime (no GPU / no model needed).

Validates the decode-loop glue on CPU/MPS:
  1. incremental centroid update == recompute-from-scratch (incl. partial blocks);
  2. AdaKV decode (torch fallback) == dense full attention at a full budget;
  3. sparse decode runs finite at a small budget and has the right shape;
  4. a multi-step decode loop stays consistent with a dense reference.

Needs torch (CPU is fine):  pip install torch
    python tests/test_runtime_cpu.py        (or: pytest -q tests/test_runtime_cpu.py)
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adakv.attention import AdaKVCache  # noqa: E402
from adakv.runtime import (  # noqa: E402
    adakv_decode_step,
    update_cache_decode,
)


def _dense_one(qh, k, v, group, D):
    """Dense attention of a single query over the full cache. qh:[Hq,D] k,v:[Hkv,S,D]."""
    kq = k.repeat_interleave(group, dim=0)
    vq = v.repeat_interleave(group, dim=0)
    a = torch.einsum("hd,hsd->hs", qh, kq) / (D ** 0.5)
    w = torch.softmax(a, dim=-1)
    return torch.einsum("hs,hsd->hd", w, vq)


def test_incremental_centroid_matches_recompute():
    torch.manual_seed(0)
    Hkv, S0, D, bs = 4, 100, 32, 16          # S0=100 -> a partial last block
    k = torch.randn(Hkv, S0, D)
    v = torch.randn(Hkv, S0, D)
    cache = AdaKVCache(block_size=bs).append_prefill(k, v)
    for _ in range(50):
        update_cache_decode(cache, torch.randn(Hkv, 1, D), torch.randn(Hkv, 1, D))
    recomputed = AdaKVCache(block_size=bs).append_prefill(cache.k, cache.v)
    diff = (cache.centroid - recomputed.centroid).abs().max()
    assert diff < 1e-4, f"incremental centroid drifted from recompute: {diff}"


def test_full_budget_matches_dense():
    torch.manual_seed(1)
    Hkv, group, S, D, bs = 4, 2, 256, 64, 16
    Hq, nb = Hkv * group, S // bs
    k = torch.randn(Hkv, S, D) * 0.1
    v = torch.randn(Hkv, S, D)
    q = torch.randn(1, Hq, 1, D)
    cache = AdaKVCache(block_size=bs).append_prefill(k, v)

    out = adakv_decode_step(q, cache, avg_budget=nb, k_min=nb, k_max=nb, backend="torch")
    dense = _dense_one(q[0, :, 0, :], k, v, group, D)
    rel = (out[0, :, 0, :] - dense).norm() / dense.norm()
    assert rel < 1e-4, f"full-budget decode != dense: rel={rel}"


def test_small_budget_runs_finite():
    torch.manual_seed(2)
    Hkv, group, S, D, bs = 8, 1, 512, 64, 16
    Hq = Hkv * group
    k = torch.randn(Hkv, S, D) * 0.1
    v = torch.randn(Hkv, S, D)
    q = torch.randn(1, Hq, 1, D)
    cache = AdaKVCache(block_size=bs).append_prefill(k, v)

    out = adakv_decode_step(q, cache, avg_budget=8, backend="torch")
    assert out.shape == (1, Hq, 1, D)
    assert torch.isfinite(out).all()


def test_multistep_decode_loop():
    torch.manual_seed(3)
    Hkv, group, S0, D, bs = 4, 2, 200, 64, 16
    Hq = Hkv * group
    k = torch.randn(Hkv, S0, D) * 0.1
    v = torch.randn(Hkv, S0, D)
    cache = AdaKVCache(block_size=bs).append_prefill(k, v)

    for _ in range(20):
        kt = torch.randn(Hkv, 1, D) * 0.1
        vt = torch.randn(Hkv, 1, D)
        update_cache_decode(cache, kt, vt)
        q = torch.randn(1, Hq, 1, D)
        nb = cache.centroid.shape[1]
        out = adakv_decode_step(q, cache, avg_budget=nb, k_min=nb, k_max=nb, backend="torch")
        dense = _dense_one(q[0, :, 0, :], cache.k, cache.v, group, D)
        rel = (out[0, :, 0, :] - dense).norm() / dense.norm()
        assert rel < 1e-4, f"step decode != dense: rel={rel}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
