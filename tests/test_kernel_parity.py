"""Triton kernel numerical parity vs the NumPy oracle.

Runs ONLY on a CUDA box with Triton. On CPU it skips (so CI/dev stays green).
This is the test that actually certifies the kernel: identical output to
adakv.reference.sparse_attention while reading only selected blocks.

    pytest -q tests/test_kernel_parity.py        # on a GPU machine
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for kernel parity", allow_module_level=True)
pytest.importorskip("triton")

from adakv.attention import AdaKVCache, adakv_decode_attention  # noqa: E402
from adakv.reference import expand_gqa, sparse_attention  # noqa: E402


@pytest.mark.parametrize("Hkv,group,S,D,bs", [(8, 4, 4096, 128, 16), (4, 1, 8192, 64, 32)])
def test_kernel_matches_oracle(Hkv, group, S, D, bs):
    torch.manual_seed(0)
    Hq = Hkv * group
    dev, dt = "cuda", torch.float16

    q = torch.randn(Hq, D, device=dev, dtype=dt)
    k = torch.randn(Hkv, S, D, device=dev, dtype=dt) * 0.1
    v = torch.randn(Hkv, S, D, device=dev, dtype=dt)

    cache = AdaKVCache(block_size=bs).append_prefill(k, v)
    out = adakv_decode_attention(q, cache, avg_budget=16, k_min=2, k_max=64)

    # Oracle on the same selection, in fp32, expanded to query heads.
    ref, _ = sparse_attention(
        q.float().cpu().numpy(),
        expand_gqa(k.float().cpu().numpy(), Hq),
        expand_gqa(v.float().cpu().numpy(), Hq),
        block_size=bs, avg_budget=16, k_min=2, k_max=64,
    )
    rel = np.linalg.norm(out.float().cpu().numpy() - ref) / (np.linalg.norm(ref) + 1e-6)
    assert rel < 2e-2, f"kernel vs oracle relative error {rel:.4f}"
