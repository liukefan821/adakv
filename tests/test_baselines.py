"""CPU tests for the H2O / SnapKV eviction baselines (NumPy oracle).

Laptop-only, no GPU. These guard the *positioning* the efficiency paper leans
on: eviction baselines fix a retained set from the PREFILL attention and can
never recover a block later, whereas AdaKV re-selects per decode query over the
full cache. The headline test (``test_eviction_loses_the_on_demand_needle``)
plants a needle that is COLD during prefill but is exactly what the decode query
wants: H2O and SnapKV evict it (recall lost), AdaKV re-selects it (recall kept)
-- at *equal budget*.

    python tests/test_baselines.py
    pytest -q tests/test_baselines.py
"""
from __future__ import annotations

import numpy as np

from adakv import baselines as B
from adakv.reference import dense_attention, sparse_attention

BS = 16
H, S, D = 4, 512, 64
NB = (S + BS - 1) // BS               # 32 blocks
SINK, LOCAL, CMIN = 1, 2, 1           # match eval_equal_budget.py defaults
FLOOR = SINK + LOCAL + CMIN           # 4
NEEDLE = 20                           # a mid content block (not sink, not local)


def _unit(rng, d):
    v = rng.standard_normal(d)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def _cold_needle_case(seed=0):
    """Prefill in which the NEEDLE block is COLD -- prefill queries have ~0
    component along the needle key direction, so it gathers little accumulated
    attention -- yet the decode query points straight at it."""
    rng = np.random.default_rng(seed)
    r = _unit(rng, D)                                        # needle direction
    K = (rng.standard_normal((H, S, D)) * 0.1).astype(np.float32)
    V = rng.standard_normal((H, S, D)).astype(np.float32)
    s = NEEDLE * BS
    K[:, s : s + BS] = r[None, None, :] * 30.0              # plant the needle
    Q = rng.standard_normal((H, S, D)).astype(np.float32)
    Q = Q - (Q @ r)[..., None] * r[None, None, :]           # project out r -> needle cold
    q_dec = np.broadcast_to(r, (H, D)).astype(np.float32).copy()   # decode query wants needle
    return Q, K, V, q_dec, r


def _adakv_mask(q, K, V, budget):
    """AdaKV (centroid, fixed budget) selection mask for decode query q."""
    _, _, mask = sparse_attention(
        q, K, V, block_size=BS, avg_budget=budget, k_min=1, k_max=NB,
        estimator="centroid", n_sink_blocks=SINK, n_local_blocks=LOCAL,
        c_min=CMIN, budget_policy="fixed", return_mask=True,
    )
    return mask


def test_shapes_and_exact_budget():
    Q, K, V, q_dec, r = _cold_needle_case()
    for mk in (B.h2o_retained_mask, B.snapkv_retained_mask):
        for budget in (6, 8, 12):
            m = mk(Q, K, block_size=BS, budget=budget,
                   n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
            assert m.shape == (H, NB)
            k = min(max(budget, FLOOR), NB)
            assert (m.sum(1) == k).all(), f"{mk.__name__} b={budget}: {m.sum(1)}"


def test_sink_and_local_always_kept():
    Q, K, V, q_dec, r = _cold_needle_case()
    for mk in (B.h2o_retained_mask, B.snapkv_retained_mask):
        m = mk(Q, K, block_size=BS, budget=8,
               n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
        assert m[:, :SINK].all(), f"{mk.__name__}: sink not forced"
        assert m[:, NB - LOCAL :].all(), f"{mk.__name__}: local not forced"


def test_h2o_keeps_prefill_heavy_hitter():
    """A block that many prefill queries attend to survives even a tight budget."""
    rng = np.random.default_rng(3)
    K = (rng.standard_normal((H, S, D)) * 0.1).astype(np.float32)
    V = rng.standard_normal((H, S, D)).astype(np.float32)
    hot, d = 11, _unit(rng, D)
    K[:, hot * BS : (hot + 1) * BS] = d[None, None, :] * 30.0
    Q = np.broadcast_to(d, (H, S, D)).astype(np.float32).copy()   # every query loves `hot`
    m = B.h2o_retained_mask(Q, K, block_size=BS, budget=6,
                            n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    assert m[:, hot].all(), "H2O should retain the global heavy hitter"


def test_eviction_loses_the_on_demand_needle():
    """HEADLINE: cold-in-prefill needle that the decode query wants.
    H2O and SnapKV evict it; AdaKV re-selects it -- all at the same budget."""
    Q, K, V, q_dec, r = _cold_needle_case(seed=0)
    budget = 8
    h2o = B.h2o_retained_mask(Q, K, block_size=BS, budget=budget,
                              n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    snap = B.snapkv_retained_mask(Q, K, block_size=BS, budget=budget,
                                  n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    ada = _adakv_mask(q_dec, K, V, budget)
    assert not h2o[:, NEEDLE].any(), "H2O should have evicted the cold needle"
    assert not snap[:, NEEDLE].any(), "SnapKV should have evicted the cold needle"
    assert ada[:, NEEDLE].all(), "AdaKV should re-select the needle for the decode query"


def test_evicted_needle_is_unreachable_in_frozen_decode():
    """Consequence of eviction: frozen decode puts ~no weight on the needle, so
    its output diverges from dense; a full retained set matches dense exactly."""
    Q, K, V, q_dec, r = _cold_needle_case(seed=1)
    dense = dense_attention(q_dec, K, V)
    h2o = B.h2o_retained_mask(Q, K, block_size=BS, budget=8,
                              n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    out_evict, _ = B.frozen_sparse_attention(q_dec, K, V, h2o, BS)
    out_full, _ = B.frozen_sparse_attention(
        q_dec, K, V, np.ones((H, NB), dtype=bool), BS)
    err_evict = np.linalg.norm(out_evict - dense) / (np.linalg.norm(dense) + 1e-9)
    err_full = np.linalg.norm(out_full - dense) / (np.linalg.norm(dense) + 1e-9)
    assert err_full < 1e-5, f"full retained set must equal dense, got {err_full}"
    assert err_evict > 0.1, f"evicting the needle must change output, got {err_evict}"


def test_frozen_selection_is_query_agnostic_unlike_adakv():
    """H2O/SnapKV retention is computed from prefill only (no decode query);
    AdaKV selection changes with the decode query."""
    Q, K, V, q_dec, r = _cold_needle_case(seed=2)
    rng = np.random.default_rng(9)
    qa = np.broadcast_to(r, (H, D)).astype(np.float32).copy()
    qb = rng.standard_normal((H, D)).astype(np.float32)
    assert not np.array_equal(_adakv_mask(qa, K, V, 8), _adakv_mask(qb, K, V, 8)), \
        "AdaKV selection should depend on the decode query"
    # the frozen baseline mask needs no decode query at all -- documents the contrast
    m = B.h2o_retained_mask(Q, K, block_size=BS, budget=8,
                            n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    assert m.sum(1).min() >= FLOOR


def test_generous_budget_frozen_matches_dense():
    Q, K, V, q_dec, r = _cold_needle_case(seed=4)
    m = B.h2o_retained_mask(Q, K, block_size=BS, budget=NB,     # retain everything
                            n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    assert m.all()
    out, _ = B.frozen_sparse_attention(q_dec, K, V, m, BS)
    dense = dense_attention(q_dec, K, V)
    err = np.linalg.norm(out - dense) / (np.linalg.norm(dense) + 1e-9)
    assert err < 1e-5, f"generous-budget frozen must equal dense, got {err}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"  ok  {n}")
    print(f"\nAll {len(tests)} baseline oracle tests passed.")
