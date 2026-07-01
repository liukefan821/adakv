"""Torch runtime for the H2O / SnapKV eviction baselines.

Mirrors ``adakv.baselines`` (the NumPy oracle) so the frozen retained set that
``patch.py`` computes on-device at prefill matches the CPU ground truth. At the
prefill call the model hands us the prompt ``(Q, K)``; we score every KV block
once (H2O: accumulated causal attention over all prompt queries; SnapKV:
attention from the last ``obs_window`` queries + key-axis max-pool) and freeze a
retained-block set that every later decode step reuses.

Everything here is **per-Q-head** (KV expanded to query heads first), matching
``adakv.reference`` and ``adakv.attention.plan_selection`` -- so H2O, SnapKV,
Quest and AdaKV all pick ``budget`` blocks per Q-head through the SAME kernel and
differ only in the scoring signal. H2O's accumulation is chunked over query
positions so it never materialises the full ``[H, L, L]`` attention matrix (that
would be ~1.7 GB at L=6K); peak memory is ``[H, chunk, L]``.

Parity with the oracle is guarded on CPU by ``tests/test_baselines_runtime.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "expand_kv_heads",
    "h2o_key_importance_torch",
    "snapkv_key_importance_torch",
    "retain_blocks_torch",
    "h2o_retained_mask_torch",
    "snapkv_retained_mask_torch",
    "mask_to_block_table",
    "frozen_decode_block_table",
]


def expand_kv_heads(x: torch.Tensor, group: int) -> torch.Tensor:
    """``[Hkv, L, D] -> [Hq, L, D]`` by repeating each KV head ``group`` times."""
    return x.repeat_interleave(group, dim=0) if group > 1 else x


@torch.no_grad()
def h2o_key_importance_torch(Q: torch.Tensor, K: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Accumulated attention each key receives over all causal prompt queries.

    ``Q, K : [H, L, D]`` (Q-head-expanded). Returns ``[H, L]`` float32. Chunked
    over query positions to bound memory to ``[H, chunk, L]``.
    """
    H, L, D = K.shape
    scale = 1.0 / math.sqrt(D)
    imp = torch.zeros(H, L, dtype=torch.float32, device=K.device)
    Kf = K.float()
    idxL = torch.arange(L, device=K.device)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        logits = torch.einsum("hcd,hld->hcl", Q[:, s:e, :].float(), Kf) * scale   # [H,c,L]
        causal = idxL.view(1, 1, -1) <= idxL[s:e].view(1, -1, 1)                   # [1,c,L]
        logits = logits.masked_fill(~causal, float("-inf"))
        imp += torch.softmax(logits, dim=-1).sum(dim=1)                            # over keys, then queries
    return imp


@torch.no_grad()
def snapkv_key_importance_torch(
    Q: torch.Tensor, K: torch.Tensor, obs_window: int = 32, pool_kernel: int = 7
) -> torch.Tensor:
    """SnapKV importance from the last ``obs_window`` queries + key-axis max-pool.

    ``Q, K : [H, L, D]``. Returns ``[H, L]`` float32. Torch ``max_pool1d`` uses
    -inf padding, which for a *max* equals the NumPy oracle's edge padding.
    """
    H, L, D = K.shape
    scale = 1.0 / math.sqrt(D)
    w = int(min(obs_window, L))
    logits = torch.einsum("hwd,hld->hwl", Q[:, L - w :, :].float(), K.float()) * scale   # [H,w,L]
    idxL = torch.arange(L, device=K.device)
    qpos = (L - w + torch.arange(w, device=K.device)).view(1, -1, 1)
    logits = logits.masked_fill(~(idxL.view(1, 1, -1) <= qpos), float("-inf"))
    imp = torch.softmax(logits, dim=-1).sum(dim=1)                                        # [H,L]
    return F.max_pool1d(imp.unsqueeze(1), kernel_size=pool_kernel, stride=1,
                        padding=pool_kernel // 2).squeeze(1)


@torch.no_grad()
def retain_blocks_torch(
    importance: torch.Tensor,
    block_size: int,
    n_real: int,
    budget: int,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
) -> torch.Tensor:
    """Retain exactly ``k = clamp(budget, floor, nb)`` blocks/head = forced sink +
    forced local + top content by ``importance``. Mirrors
    ``adakv.baselines.retain_blocks`` / ``adakv.selector`` accounting.

    ``importance : [H, L]``. Returns mask ``[H, nb]`` bool.
    """
    H, L = importance.shape
    nb = (n_real + block_size - 1) // block_size
    imp = importance[:, :n_real]
    pad = nb * block_size - n_real
    if pad:
        imp = F.pad(imp, (0, pad), value=0.0)
    blk = imp.view(H, nb, block_size).sum(dim=2)                    # [H, nb]

    sink, local = min(n_sink_blocks, nb), min(n_local_blocks, nb)
    biased = blk.clone()
    if sink:
        biased[:, :sink] = float("inf")
    if local:
        biased[:, nb - local :] = float("inf")

    floor = min(sink + local + c_min, nb)
    k = min(max(int(budget), floor), nb)
    order = biased.argsort(dim=-1, descending=True)
    mask = torch.zeros(H, nb, dtype=torch.bool, device=importance.device)
    mask.scatter_(1, order[:, :k], True)
    if sink:
        mask[:, :sink] = True
    if local:
        mask[:, nb - local :] = True
    return mask


def h2o_retained_mask_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    block_size: int = 16,
    budget: int = 8,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
    chunk: int = 256,
) -> torch.Tensor:
    """Frozen H2O retained-block mask ``[H, nb]`` from the prompt (Q, K)."""
    imp = h2o_key_importance_torch(Q, K, chunk=chunk)
    return retain_blocks_torch(imp, block_size, K.shape[1], budget,
                               n_sink_blocks, n_local_blocks, c_min)


def snapkv_retained_mask_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    block_size: int = 16,
    budget: int = 8,
    obs_window: int = 32,
    pool_kernel: int = 7,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
) -> torch.Tensor:
    """Frozen SnapKV retained-block mask ``[H, nb]`` from the prompt (Q, K)."""
    imp = snapkv_key_importance_torch(Q, K, obs_window, pool_kernel)
    return retain_blocks_torch(imp, block_size, K.shape[1], budget,
                               n_sink_blocks, n_local_blocks, c_min)


def mask_to_block_table(mask: torch.Tensor):
    """Convert a fixed-budget retained mask ``[H, nb]`` (equal count per head) to
    the kernel's ``(block_table [H, k] int32, sel_lens [H] int32)`` format.

    Every head retains exactly ``k`` blocks (fixed budget), so the True positions
    of each row -- returned by ``nonzero`` in ascending block order -- reshape to
    ``[H, k]`` directly.
    """
    H, nb = mask.shape
    counts = mask.sum(dim=1)
    k = int(counts.max().item())
    assert bool((counts == k).all()), "mask_to_block_table expects an equal count per head"
    block_table = mask.nonzero(as_tuple=False)[:, 1].view(H, k).to(torch.int32).contiguous()
    return block_table, counts.to(torch.int32)


@torch.no_grad()
def frozen_decode_block_table(
    fmask: torch.Tensor, nb_now: int, n_sink_blocks: int, n_local_blocks: int
):
    """Decode-step ``(block_table, sel_lens)`` from the frozen prefill mask.

    ``fmask : [H, nb_prefill]`` bool -- the retained set fixed at prefill. As
    decode extends the sequence to ``nb_now >= nb_prefill`` blocks, the frozen
    CONTENT selection never changes (that is the eviction), and only the sink +
    local (recent) stabilisers track the current tail. So newly generated blocks
    are attended while evicted content stays evicted -- the query-agnostic,
    lossy behaviour H2O/SnapKV are meant to model.

    Per-head counts stay equal (the current local window can only reach back into
    the prefill-local blocks, which were already retained), so the result is
    kernel-ready. Realized budget = frozen budget + new trailing blocks, usually
    +0..2 over a short answer; the budget trace records it honestly.
    """
    H, nbp = fmask.shape
    assert nb_now >= nbp, "decode block count must not shrink below prefill"
    sink = min(n_sink_blocks, nb_now)
    local = min(n_local_blocks, nb_now)
    dmask = torch.zeros(H, nb_now, dtype=torch.bool, device=fmask.device)
    dmask[:, :nbp] = fmask
    if sink:
        dmask[:, :sink] = True
    if local:
        dmask[:, nb_now - local :] = True
    return mask_to_block_table(dmask)
