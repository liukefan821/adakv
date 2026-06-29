"""CPU correctness tests for the AdaKV algorithm (NumPy oracle).

These validate *algorithmic* behaviour, not the kernel:
  1. shapes / GQA expansion;
  2. forced sink + local blocks are always selected;
  3. a planted high-relevance block is recovered;
  4. with a generous budget the sparse output converges to dense attention;
  5. adaptive budget responds to score concentration.

Kernel numerical parity (Triton vs this oracle) lives in test_kernel_parity.py
and runs only on a CUDA box.

Run directly:  python tests/test_reference.py
Or:            pytest -q tests/test_reference.py
"""
from __future__ import annotations

import numpy as np

from adakv.budget import allocate_budget
from adakv.reference import dense_attention, expand_gqa, sparse_attention


def _make_case(H=4, S=512, D=64, planted_block=7, block_size=16, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((H, D)).astype(np.float32)
    K = rng.standard_normal((H, S, D)).astype(np.float32) * 0.1
    V = rng.standard_normal((H, S, D)).astype(np.float32)
    # Plant a far-away block strongly aligned with q for every head.
    s = planted_block * block_size
    K[:, s : s + block_size] = q[:, None, :] * 3.0
    return q, K, V


def test_shapes_and_gqa():
    H_kv, S, D = 2, 256, 32
    rng = np.random.default_rng(1)
    K_kv = rng.standard_normal((H_kv, S, D)).astype(np.float32)
    K_q = expand_gqa(K_kv, 8)
    assert K_q.shape == (8, S, D)
    # group repeat: heads 0..3 share kv-head 0, 4..7 share kv-head 1
    assert np.allclose(K_q[0], K_q[3]) and np.allclose(K_q[4], K_q[7])
    assert not np.allclose(K_q[3], K_q[4])

    q = rng.standard_normal((8, D)).astype(np.float32)
    out, counts = sparse_attention(q, K_q, expand_gqa(K_kv, 8), block_size=16)
    assert out.shape == (8, D)
    assert counts.shape == (8,)


def test_forced_blocks_selected():
    q, K, V = _make_case()
    _, _, mask = sparse_attention(
        q, K, V, block_size=16, avg_budget=6, n_sink_blocks=2, n_local_blocks=3,
        return_mask=True,
    )
    B = mask.shape[1]
    assert mask[:, :2].all(), "sink blocks must always be selected"
    assert mask[:, B - 3 :].all(), "local window must always be selected"


def test_planted_block_recovered():
    q, K, V = _make_case(planted_block=7, block_size=16)
    _, _, mask = sparse_attention(
        q, K, V, block_size=16, avg_budget=6, n_sink_blocks=1, n_local_blocks=2,
        return_mask=True,
    )
    assert mask[:, 7].all(), "the strongly-aligned block must be selected by every head"


def test_converges_to_dense():
    q, K, V = _make_case(S=512, block_size=16)
    n_blocks = 512 // 16
    dense = dense_attention(q, K, V)
    # Budget = all blocks -> sparse must equal dense.
    sp_full, _ = sparse_attention(
        q, K, V, block_size=16, avg_budget=n_blocks, k_min=n_blocks, k_max=n_blocks,
    )
    assert np.allclose(dense, sp_full, atol=1e-4), np.abs(dense - sp_full).max()

    # Small budget but planted block present -> close to dense (mass concentrated).
    sp_small, counts = sparse_attention(q, K, V, block_size=16, avg_budget=6)
    rel = np.linalg.norm(dense - sp_small) / (np.linalg.norm(dense) + 1e-6)
    assert rel < 0.05, f"relative error {rel:.4f} too high at small budget"
    assert counts.max() <= n_blocks


def test_adaptive_budget_responds_to_concentration():
    # Budget is allocated *relative* to other heads in the same call (the mean is
    # pinned to avg_budget), so put peaked and flat heads together and check the
    # budget flows from the peaked heads to the flat ones.
    H, B = 8, 64
    rng = np.random.default_rng(3)
    scores = (rng.standard_normal((H, B)) * 0.01).astype(np.float32)  # ~uniform base
    scores[:4, 0] += 30.0  # first 4 heads sharply peaked; last 4 stay flat

    k = allocate_budget(scores, avg_budget=8, k_min=2, k_max=48)

    assert k[:4].mean() < k[4:].mean(), "flat heads should get more budget than peaked"
    assert k.min() >= 2 and k.max() <= 48


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
