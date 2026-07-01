"""CPU parity: torch H2O/SnapKV runtime selection == the NumPy oracle.

Laptop-only, torch on CPU (no GPU, no model). Same contract as the Phase-2
vectorisation guard (torch selection must agree with the NumPy oracle): the
on-device frozen-retention math in ``adakv.baselines_rt`` must reproduce the
ground truth in ``adakv.baselines``, so what runs on the T4 at prefill is exactly
what these CPU tests certify.

    python tests/test_baselines_runtime.py
    pytest -q tests/test_baselines_runtime.py
"""
from __future__ import annotations

import numpy as np
import torch

from adakv import baselines as B          # NumPy oracle
from adakv import baselines_rt as RT      # torch runtime
from adakv.attention import (
    record_selection, reset_budget_trace, get_budget_trace, stop_budget_trace,
    set_recall_target, get_recall, clear_recall_target,
)

BS = 16
H, S, D = 6, 384, 64
NB = (S + BS - 1) // BS
SINK, LOCAL, CMIN = 1, 2, 1
BUDGETS = (6, 8, 12)


def _qk(seed=0):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((H, S, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, S, D)) * 0.5).astype(np.float32)
    return Q, K, torch.from_numpy(Q), torch.from_numpy(K)


def _agree(np_mask, torch_mask):
    return float((np_mask == torch_mask.cpu().numpy()).mean())


def test_h2o_importance_close():
    """Chunked torch accumulation matches the full-matrix NumPy accumulation."""
    for seed in (0, 1):
        Qn, Kn, Qt, Kt = _qk(seed)
        inp = B.h2o_key_importance(Qn, Kn)
        it = RT.h2o_key_importance_torch(Qt, Kt, chunk=64).cpu().numpy()   # small chunk on purpose
        rel = np.abs(it - inp).max() / (np.abs(inp).max() + 1e-9)
        assert rel < 1e-4, f"h2o importance rel err {rel:.2e}"


def test_snapkv_importance_close():
    for seed in (0, 1):
        Qn, Kn, Qt, Kt = _qk(seed)
        inp = B.snapkv_key_importance(Qn, Kn, obs_window=32, pool_kernel=7)
        it = RT.snapkv_key_importance_torch(Qt, Kt, obs_window=32, pool_kernel=7).cpu().numpy()
        rel = np.abs(it - inp).max() / (np.abs(inp).max() + 1e-9)
        assert rel < 1e-4, f"snapkv importance rel err {rel:.2e}"


def test_h2o_mask_parity():
    for seed in (0, 1, 2):
        Qn, Kn, Qt, Kt = _qk(seed)
        for b in BUDGETS:
            nm = B.h2o_retained_mask(Qn, Kn, block_size=BS, budget=b,
                                     n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
            tm = RT.h2o_retained_mask_torch(Qt, Kt, block_size=BS, budget=b,
                                            n_sink_blocks=SINK, n_local_blocks=LOCAL,
                                            c_min=CMIN, chunk=64)
            assert _agree(nm, tm) > 0.99, f"H2O seed={seed} b={b}: agree {_agree(nm, tm):.3f}"


def test_snapkv_mask_parity():
    for seed in (0, 1, 2):
        Qn, Kn, Qt, Kt = _qk(seed)
        for b in BUDGETS:
            nm = B.snapkv_retained_mask(Qn, Kn, block_size=BS, budget=b,
                                        n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
            tm = RT.snapkv_retained_mask_torch(Qt, Kt, block_size=BS, budget=b,
                                               n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
            assert _agree(nm, tm) > 0.99, f"SnapKV seed={seed} b={b}: agree {_agree(nm, tm):.3f}"


def test_exact_budget_torch():
    Qn, Kn, Qt, Kt = _qk(3)
    for mk in (RT.h2o_retained_mask_torch, RT.snapkv_retained_mask_torch):
        for b in BUDGETS:
            m = mk(Qt, Kt, block_size=BS, budget=b, n_sink_blocks=SINK,
                   n_local_blocks=LOCAL, c_min=CMIN)
            k = min(max(b, SINK + LOCAL + CMIN), NB)
            assert bool((m.sum(1) == k).all()), f"{mk.__name__} b={b}: counts {m.sum(1).tolist()}"


def test_mask_to_block_table():
    Qn, Kn, Qt, Kt = _qk(4)
    m = RT.h2o_retained_mask_torch(Qt, Kt, block_size=BS, budget=8,
                                   n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    bt, sl = RT.mask_to_block_table(m)
    k = min(max(8, SINK + LOCAL + CMIN), NB)
    assert bt.shape == (H, k) and sl.shape == (H,)
    assert bool((sl == k).all())
    for h in range(H):
        expected = torch.nonzero(m[h], as_tuple=False).flatten().to(torch.int32)
        assert torch.equal(bt[h], expected), f"head {h} block_table mismatch"


def test_gqa_expand():
    Hkv, group = 2, 3
    x = torch.randn(Hkv, S, D)
    xe = RT.expand_kv_heads(x, group)
    assert xe.shape == (Hkv * group, S, D)
    for g in range(group):
        assert torch.equal(xe[g], x[0]) and torch.equal(xe[group + g], x[1])


def test_frozen_decode_block_table():
    """Decode-time reconstruction: content selection frozen, recent window tracks
    the growing tail, per-head counts stay equal."""
    _, _, Qt, Kt = _qk(5)
    fmask = RT.h2o_retained_mask_torch(Qt, Kt, block_size=BS, budget=8,
                                       n_sink_blocks=SINK, n_local_blocks=LOCAL, c_min=CMIN)
    Hf, nbp = fmask.shape
    k = int(fmask.sum(1)[0])

    # no growth -> exactly the frozen set (sink/local were already forced)
    bt0, sl0 = RT.frozen_decode_block_table(fmask, nbp, SINK, LOCAL)
    dm0 = torch.zeros(Hf, nbp, dtype=torch.bool)
    dm0.scatter_(1, bt0.long(), True)
    assert torch.equal(dm0, fmask) and bool((sl0 == k).all())

    # grow by 3 blocks
    grow, nb_now = 3, nbp + 3
    bt, sl = RT.frozen_decode_block_table(fmask, nb_now, SINK, LOCAL)
    dm = torch.zeros(Hf, nb_now, dtype=torch.bool)
    dm.scatter_(1, bt.long(), True)
    assert torch.equal(dm[:, SINK : nbp - LOCAL], fmask[:, SINK : nbp - LOCAL]), \
        "evicted content must stay evicted (frozen)"
    assert bool(dm[:, nb_now - LOCAL :].all()), "current local window must be attended"
    assert bool((sl == k + min(LOCAL, grow)).all()), f"counts drifted: {sl.tolist()}"


def test_record_selection_hooks():
    """record_selection drives the same budget-trace + recall instruments the
    harness reads, so the frozen path reports metrics like plan_selection."""
    reset_budget_trace()
    set_recall_target([2, 5])
    bt = torch.tensor([[2, 4, 7], [0, 5, 9]], dtype=torch.int32)   # H=2, 3 blocks each
    sl = torch.tensor([3, 3], dtype=torch.int32)
    record_selection(bt, sl, nb=10)
    tr = get_budget_trace(); stop_budget_trace()
    r = get_recall(); clear_recall_target()
    assert len(tr) == 1 and abs(tr[0] - 3.0) < 1e-6, f"budget trace {tr}"
    assert abs(r - 1.0) < 1e-6, f"both targets covered -> recall 1.0, got {r}"


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"  ok  {n}")
    print(f"\nAll {len(tests)} runtime-parity tests passed.")
