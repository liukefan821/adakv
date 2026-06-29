"""Triton kernel numerical parity vs a same-selection fp32 reference.

Isolates the KERNEL: compute the block selection once (host), run the kernel on
it, and compare against an fp32 reference that attends to the *same* selected
blocks. This certifies the kernel computes correct masked attention, separate
from any host-vs-oracle selection drift.

Runs ONLY on a CUDA box with Triton; skips on CPU.
    pytest -q tests/test_kernel_parity.py -s
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for kernel parity", allow_module_level=True)
pytest.importorskip("triton")

from adakv.attention import AdaKVCache, plan_selection  # noqa: E402
from adakv.kernels.block_sparse_decode import block_sparse_decode  # noqa: E402


def _ref_from_table(q, k, v, block_table, sel_lens, block_size, group):
    """fp32 reference: attend to exactly the kernel's selected blocks."""
    Hq, D = q.shape
    Hkv, S, _ = k.shape
    out = np.zeros((Hq, D), dtype=np.float32)
    scale = 1.0 / np.sqrt(D)
    for h in range(Hq):
        kvh = h // group
        idx = []
        for j in range(int(sel_lens[h])):
            b = int(block_table[h, j])
            idx.extend(range(b * block_size, min((b + 1) * block_size, S)))
        idx = np.array(sorted(set(idx)), dtype=np.int64)
        kk, vv = k[kvh, idx], v[kvh, idx]
        s = (kk @ q[h]) * scale
        w = np.exp(s - s.max())
        w /= w.sum()
        out[h] = w @ vv
    return out


@pytest.mark.parametrize("Hkv,group,S,D,bs", [(8, 4, 4096, 128, 16), (4, 1, 8192, 64, 32)])
def test_kernel_matches_reference(Hkv, group, S, D, bs):
    torch.manual_seed(0)
    Hq = Hkv * group
    dev, dt = "cuda", torch.float16

    q = torch.randn(Hq, D, device=dev, dtype=dt)
    k = torch.randn(Hkv, S, D, device=dev, dtype=dt) * 0.1
    v = torch.randn(Hkv, S, D, device=dev, dtype=dt)

    cache = AdaKVCache(block_size=bs).append_prefill(k, v)
    block_table, sel_lens = plan_selection(q, cache, avg_budget=16, k_min=2, k_max=64)

    out = block_sparse_decode(q, cache.k, cache.v, block_table, sel_lens, bs)

    ref = _ref_from_table(
        q.float().cpu().numpy(), k.float().cpu().numpy(), v.float().cpu().numpy(),
        block_table.cpu().numpy(), sel_lens.cpu().numpy(), bs, group,
    )
    rel = np.linalg.norm(out.float().cpu().numpy() - ref) / (np.linalg.norm(ref) + 1e-6)
    print(f"[Hkv={Hkv} group={group} S={S} D={D} bs={bs}] "
          f"rel_err={rel:.3e}  sel_lens(mean={sel_lens.float().mean():.1f}, "
          f"min={int(sel_lens.min())}, max={int(sel_lens.max())})")
    assert rel < 2e-2, f"kernel vs reference relative error {rel:.4f}"
