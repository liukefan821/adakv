"""CPU tests for the (estimator x budget_policy) selection factorial.

Runs on a laptop with no GPU. Guards the Phase-2 selection logic:
  - all grid cells run (fixed / adaptive-entropy / adaptive-nucleus x centroid / minmax);
  - no head is ever starved below c_min content blocks (the collapse bug);
  - fixed budget is honored; adaptive targets the mean;
  - torch selection matches the NumPy oracle.

    pytest -q tests/test_factorial_selection.py
"""
import numpy as np
import torch

from adakv.attention import (
    AdaKVCache, plan_selection,
    reset_budget_trace, get_budget_trace, stop_budget_trace,
    set_recall_target, get_recall, clear_recall_target,
)
from adakv import reference as R

BS = 16
HKV, S, D, GROUP = 2, 320, 64, 6   # 12 q-heads, GQA group 6, 20 blocks
SINK, LOCAL, CMIN = 1, 2, 1
FLOOR = SINK + LOCAL + CMIN
POLICIES = ["fixed", "adaptive", "adaptive_nucleus"]


def _fixtures(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Hq = HKV * GROUP
    k = torch.randn(HKV, S, D); v = torch.randn(HKV, S, D); q = torch.randn(Hq, D)
    cc = AdaKVCache(block_size=BS, estimator="centroid").append_prefill(k, v)
    cm = AdaKVCache(block_size=BS, estimator="minmax").append_prefill(k, v)
    return q, k, v, cc, cm, Hq


def _plan(q, cache, est, pol, avg_budget=8):
    return plan_selection(q, cache, avg_budget=avg_budget, estimator=est,
                          budget_policy=pol, n_sink_blocks=SINK, n_local_blocks=LOCAL,
                          c_min=CMIN, k_max=256)


def test_all_six_cells_run():
    q, k, v, cc, cm, Hq = _fixtures()
    for est, cache in [("centroid", cc), ("minmax", cm)]:
        for pol in POLICIES:
            bt, sl = _plan(q, cache, est, pol)
            assert bt.shape[0] == Hq and sl.shape[0] == Hq
            assert int(sl.max()) <= cache.centroid.shape[1]


def test_no_collapse_every_head_keeps_content():
    """The Phase-2 fix: every head keeps >= c_min content blocks (sel - stab >= c_min)."""
    q, k, v, cc, cm, Hq = _fixtures()
    stab = SINK + LOCAL
    for est, cache in [("centroid", cc), ("minmax", cm)]:
        for pol in POLICIES:
            for ab in [4, 6, 8]:
                _, sl = _plan(q, cache, est, pol, avg_budget=ab)
                content = sl.numpy() - stab
                assert content.min() >= CMIN, f"{est}/{pol}/ab={ab}: starved, content={content.min()}"


def test_fixed_budget_is_honored():
    q, k, v, cc, cm, Hq = _fixtures()
    for est, cache in [("centroid", cc), ("minmax", cm)]:
        _, sl = _plan(q, cache, est, "fixed", avg_budget=10)
        assert (sl.numpy() == 10).all()


def test_adaptive_targets_mean_budget():
    q, k, v, cc, cm, Hq = _fixtures()
    for pol in ["adaptive", "adaptive_nucleus"]:
        _, sl = _plan(q, cc, "centroid", pol, avg_budget=8)
        m = float(sl.float().mean())
        assert FLOOR <= m <= 8 + 2.0, f"{pol}: realized mean {m} off target"


def test_minmax_needs_minmax_cache():
    q, k, v, cc, cm, Hq = _fixtures()
    try:
        _plan(q, cc, "minmax", "fixed")
        raise AssertionError("expected ValueError for minmax estimator on centroid cache")
    except ValueError:
        pass


def test_torch_matches_numpy_oracle():
    """torch selection must agree with the NumPy oracle for matched settings."""
    q, k, v, cc, cm, Hq = _fixtures()
    kq = np.repeat(k.numpy(), GROUP, 0); vq = np.repeat(v.numpy(), GROUP, 0); qn = q.numpy()
    for est, cache in [("centroid", cc), ("minmax", cm)]:
        for pol in POLICIES:
            _, _, mask = R.sparse_attention(
                qn, kq, vq, block_size=BS, avg_budget=8, k_min=2, k_max=256,
                estimator=est, n_sink_blocks=SINK, n_local_blocks=LOCAL,
                c_min=CMIN, budget_policy=pol, return_mask=True)
            bt, sl = _plan(q, cache, est, pol)
            nb = cache.centroid.shape[1]
            tmask = np.zeros((Hq, nb), dtype=bool)
            for h in range(Hq):
                tmask[h, bt[h, :int(sl[h])].numpy()] = True
            agree = (tmask == mask).mean()
            assert agree > 0.99, f"{est}/{pol}: torch-vs-numpy agreement only {agree:.3f}"


def test_recall_instrumentation():
    q, k, v, cc, cm, Hq = _fixtures()
    set_recall_target([SINK])           # a forced sink block -> always covered
    for _ in range(3):
        _plan(q, cc, "centroid", "adaptive")
    r = get_recall(); clear_recall_target()
    assert 0.0 <= r <= 1.0


def test_budget_trace():
    q, k, v, cc, cm, Hq = _fixtures()
    reset_budget_trace()
    for _ in range(3):
        _plan(q, cc, "centroid", "adaptive")
    tr = get_budget_trace(); stop_budget_trace()
    assert len(tr) == 3 and all(t > 0 for t in tr)
