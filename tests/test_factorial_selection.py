"""CPU tests for the (estimator x budget_policy) selection factorial.

Runs on a laptop with no GPU. Guards the Phase-2 generalisation of
plan_selection: AdaKV (centroid+adaptive) must be unchanged, Quest
(minmax+fixed) and the two ablation cells must run, fixed budget must be
honored, and torch selection must match the NumPy oracle.

    pytest -q tests/test_factorial_selection.py
"""
import numpy as np
import torch

from adakv.attention import (
    AdaKVCache, plan_selection,
    reset_budget_trace, get_budget_trace, stop_budget_trace,
)
from adakv import reference as R
from adakv.estimator import build_block_summaries, block_scores

BS = 16
HKV, S, D, GROUP = 2, 320, 64, 6   # 12 q-heads, GQA group 6, 20 blocks


def _fixtures(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Hq = HKV * GROUP
    k = torch.randn(HKV, S, D); v = torch.randn(HKV, S, D); q = torch.randn(Hq, D)
    cc = AdaKVCache(block_size=BS, estimator="centroid").append_prefill(k, v)
    cm = AdaKVCache(block_size=BS, estimator="minmax").append_prefill(k, v)
    return q, k, v, cc, cm, Hq


def test_all_four_cells_run():
    q, k, v, cc, cm, Hq = _fixtures()
    for cache, est, pol in [(cc, "centroid", "adaptive"), (cm, "minmax", "fixed"),
                            (cc, "centroid", "fixed"), (cm, "minmax", "adaptive")]:
        bt, sl = plan_selection(q, cache, avg_budget=8, estimator=est, budget_policy=pol)
        assert bt.shape[0] == Hq and sl.shape[0] == Hq
        assert int(sl.min()) >= 1 and int(sl.max()) <= cache.centroid.shape[1]


def test_fixed_budget_is_honored():
    """Above the sink+local floor, every head gets exactly avg_budget."""
    q, k, v, cc, cm, Hq = _fixtures()
    for cache, est in [(cc, "centroid"), (cm, "minmax")]:
        _, sl = plan_selection(q, cache, avg_budget=10, estimator=est,
                               budget_policy="fixed", n_sink_blocks=1, n_local_blocks=4)
        assert (sl.numpy() == 10).all()


def test_adaptive_targets_mean_budget():
    q, k, v, cc, cm, Hq = _fixtures()
    _, sl = plan_selection(q, cc, avg_budget=8, estimator="centroid", budget_policy="adaptive")
    assert abs(float(sl.float().mean()) - 8.0) <= 1.5  # rounding + floor slack


def test_minmax_needs_minmax_cache():
    q, k, v, cc, cm, Hq = _fixtures()
    try:
        plan_selection(q, cc, avg_budget=8, estimator="minmax", budget_policy="fixed")
        raise AssertionError("expected ValueError for minmax estimator on centroid cache")
    except ValueError:
        pass


def test_torch_matches_numpy_oracle_centroid_adaptive():
    """The validated AdaKV path must agree with the pure-NumPy oracle selection."""
    q, k, v, cc, cm, Hq = _fixtures()
    kq = np.repeat(k.numpy(), GROUP, 0); vq = np.repeat(v.numpy(), GROUP, 0); qn = q.numpy()
    _, _, mask = R.sparse_attention(qn, kq, vq, block_size=BS, avg_budget=8,
                                    k_min=2, k_max=64, estimator="centroid", return_mask=True)
    bt, sl = plan_selection(q, cc, avg_budget=8, estimator="centroid", budget_policy="adaptive")
    nb = cc.centroid.shape[1]
    tmask = np.zeros((Hq, nb), dtype=bool)
    for h in range(Hq):
        tmask[h, bt[h, :int(sl[h])].numpy()] = True
    assert (tmask == mask).all()


def test_minmax_scores_match_numpy():
    q, k, v, cc, cm, Hq = _fixtures()
    kq = np.repeat(k.numpy(), GROUP, 0); qn = q.numpy()
    sc_np = block_scores(qn, build_block_summaries(kq, BS, "minmax"))
    kmn = cm.kmin.repeat_interleave(GROUP, 0).float()
    kmx = cm.kmax.repeat_interleave(GROUP, 0).float()
    qf = q.float()
    sc_t = (qf.clamp_min(0)[:, None, :] * kmx + qf.clamp_max(0)[:, None, :] * kmn).sum(-1).numpy()
    assert np.abs(sc_t - sc_np).max() < 1e-3


def test_budget_trace():
    q, k, v, cc, cm, Hq = _fixtures()
    reset_budget_trace()
    for _ in range(3):
        plan_selection(q, cc, avg_budget=8, estimator="centroid", budget_policy="adaptive")
    tr = get_budget_trace(); stop_budget_trace()
    assert len(tr) == 3 and all(t > 0 for t in tr)
