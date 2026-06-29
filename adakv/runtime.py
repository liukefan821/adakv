"""End-to-end decode runtime for AdaKV (model-agnostic, CPU/GPU).

Sits between a model's attention call and the AdaKV pieces:
  - prefill: dense causal attention; ``AdaKVCache.append_prefill`` builds the
    block-summary cache.
  - decode:  per step, ``update_cache_decode`` appends the new KV and refreshes
    the affected block centroid in O(D); ``adakv_decode_step`` plans the per-head
    block selection and runs attention over only the selected blocks.

The decode-attention backend is chosen automatically:
  - CUDA + Triton  -> the fused block-sparse kernel (kernels.block_sparse_decode)
  - otherwise      -> a pure-torch masked reference (CPU/MPS), so the whole
                      integration is testable on a laptop with no GPU.

Layout: [B, H, L, D] (HuggingFace attention layout). Batch is squeezed to 1 for
now (single-sequence eval, e.g. needle); B > 1 is a TODO (loop over batch).
"""
from __future__ import annotations

import torch

from .attention import AdaKVCache, plan_selection
from .kernels.block_sparse_decode import HAS_TRITON, block_sparse_decode


def dense_prefill(q, k, v, causal: bool = True):
    """Dense scaled-dot-product attention for prefill.

    q: [B, Hq, L, D]; k, v: [B, Hkv, L, D] (GQA expanded internally). -> [B, Hq, L, D]
    """
    B, Hq, L, D = q.shape
    g = Hq // k.shape[1]
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    a = torch.einsum("bhld,bhsd->bhls", q.float(), k.float()) / (D ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        a = a.masked_fill(mask, float("-inf"))
    w = torch.softmax(a, dim=-1)
    return torch.einsum("bhls,bhsd->bhld", w, v.float()).to(q.dtype)


def update_cache_decode(cache: AdaKVCache, k_t, v_t):
    """Append one decoded token's KV and refresh the affected block centroid.

    k_t, v_t: [Hkv, 1, D]. Incremental running-mean update is O(D) per step.
    """
    S_old = cache.k.shape[1]
    cache.k = torch.cat([cache.k, k_t], dim=1)
    cache.v = torch.cat([cache.v, v_t], dim=1)

    b = S_old // cache.block_size          # block the new token lands in
    cnt_before = S_old - b * cache.block_size
    x = k_t[:, 0, :]                        # [Hkv, D]
    if b >= cache.centroid.shape[1]:        # new block opens
        cache.centroid = torch.cat([cache.centroid, x.unsqueeze(1)], dim=1)
    else:                                   # running mean of the active block
        mu = cache.centroid[:, b, :]
        cache.centroid[:, b, :] = mu + (x - mu) / (cnt_before + 1)
    return cache


def _torch_masked_decode(q, k, v, block_table, sel_lens, block_size, n_blocks):
    """Pure-torch CPU fallback: attention over exactly the selected blocks.

    Mirrors the Triton kernel's semantics. q: [Hq, D]; k, v: [Hkv, S, D].
    """
    Hq, D = q.shape
    Hkv, S, _ = k.shape
    g = Hq // Hkv
    kq = k.repeat_interleave(g, dim=0)
    vq = v.repeat_interleave(g, dim=0)

    bm = torch.zeros(Hq, n_blocks, dtype=torch.bool, device=q.device)
    for h in range(Hq):
        bm[h, block_table[h, : int(sel_lens[h])].long()] = True
    tok = bm.repeat_interleave(block_size, dim=1)[:, :S]  # [Hq, S]

    a = torch.einsum("hd,hsd->hs", q.float(), kq.float()) / (D ** 0.5)
    a = a.masked_fill(~tok, float("-inf"))
    w = torch.softmax(a, dim=-1)
    return torch.einsum("hs,hsd->hd", w, vq.float()).to(q.dtype)


def adakv_decode_step(
    q,                      # [1, Hq, 1, D]
    cache: AdaKVCache,
    avg_budget: int,
    k_min: int = 2,
    k_max: int = 64,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    temperature: float = 1.0,
    backend: str = "auto",   # "auto" | "triton" | "torch"
):
    """One AdaKV decode step. Returns [1, Hq, 1, D] (HF attention layout)."""
    assert q.shape[0] == 1, "runtime currently supports batch size 1 (B>1 is a TODO)"
    qh = q[0, :, 0, :]       # [Hq, D]

    block_table, sel_lens = plan_selection(
        qh, cache, avg_budget, k_min, k_max, n_sink_blocks, n_local_blocks, temperature
    )

    use_triton = backend == "triton" or (backend == "auto" and HAS_TRITON and qh.is_cuda)
    if use_triton:
        out = block_sparse_decode(qh, cache.k, cache.v, block_table, sel_lens, cache.block_size)
    else:
        out = _torch_masked_decode(
            qh, cache.k, cache.v, block_table, sel_lens,
            cache.block_size, cache.centroid.shape[1],
        )
    return out.unsqueeze(0).unsqueeze(2)  # [1, Hq, 1, D]
