"""Eviction-baseline oracles (H2O, SnapKV) in pure NumPy.

These are the *lossy, decode-time query-agnostic* baselines that AdaKV/Quest are
contrasted against. They decide **once**, from the prefill attention, which KV
blocks to retain inside a fixed budget, then attend to that **frozen** set for
every decode step. AdaKV/Quest (see ``adakv.reference``) instead keep the full
cache and **re-select** blocks per decode query. Adding these baselines lets the
paper show, at *equal budget*, that permanently discarding blocks costs the
retrieval recall that per-step re-selection preserves.

Faithful **block-level** analogues -- same block granularity, and downstream the
same gather/kernel, so an equal-budget comparison isolates *evict vs re-select*
rather than *token-vs-block* or *kernel A vs kernel B*:

- **H2O**  : retain blocks with the largest *accumulated* attention over ALL
  causal prefill query positions (global heavy hitters). Ref: Zhang et al.,
  "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs",
  NeurIPS 2023.
- **SnapKV**: retain blocks most attended by an *observation window* (the last
  ``obs_window`` prefill queries), with 1-D max-pooling over key positions to
  prefer contiguous clusters (SnapKV's signature). Ref: Li et al., "SnapKV: LLM
  Knows What You are Looking for Before Generation", NeurIPS 2024.

Both are LOSSY: a block outside the retained set is gone for the rest of decode.
Both are query-agnostic at decode time: the retained set is fixed once, so a
fact the decode query later needs is unreachable if it was not a prefill
heavy-hitter / not attended by the observation window -- exactly the failure the
paper attributes to eviction.

Layout convention (matches ``adakv.reference``): ``[n_heads, seq_len, head_dim]``
already expanded to query heads. GQA/MQA (per-KV-head cache compression) is
handled by the runtime caller, not here; this module defines the per-head
algorithmic semantics and is the CPU-testable ground truth.

Complexity note: H2O importance forms the full ``[H, S, S]`` prefill attention
(O(H*S^2)); this is fine for the oracle/tests and is a *one-shot* end-of-prefill
computation at runtime (chunk it there for very long contexts). The decode step
itself only ever touches the retained blocks.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "h2o_key_importance",
    "snapkv_key_importance",
    "retain_blocks",
    "h2o_retained_mask",
    "snapkv_retained_mask",
    "frozen_sparse_attention",
]


def _scale(D: int) -> float:
    return 1.0 / np.sqrt(D)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# --- per-key importance signals ---------------------------------------------
def h2o_key_importance(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Accumulated attention each key receives over all causal prefill queries.

    H2O's "heavy hitter" signal. ``Q, K : [H, S, D]`` (prefill queries and keys,
    expanded to query heads). Returns ``[H, S]``.
    """
    H, S, D = K.shape
    assert Q.shape == (H, S, D), "Q and K must share [H, S, D]"
    logits = np.einsum("hid,hjd->hij", Q.astype(np.float64), K.astype(np.float64)) * _scale(D)
    # causal: query i (row) may attend key j (col) only if j <= i.
    idx = np.arange(S)
    causal = idx[None, :] <= idx[:, None]          # [S, S] True where j <= i
    logits = np.where(causal[None], logits, -np.inf)
    w = _softmax(logits, axis=2)                   # softmax over keys, per query
    return w.sum(axis=1).astype(np.float32)        # sum over queries -> [H, S]


def _maxpool1d_same(x: np.ndarray, kernel: int) -> np.ndarray:
    """Same-length 1-D max pool over the last axis (odd kernel)."""
    if kernel <= 1:
        return x
    H, S = x.shape
    pad = kernel // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(x)
    for t in range(S):
        out[:, t] = xp[:, t : t + kernel].max(axis=1)
    return out


def snapkv_key_importance(
    Q: np.ndarray, K: np.ndarray, obs_window: int = 32, pool_kernel: int = 7
) -> np.ndarray:
    """SnapKV importance: attention from the last ``obs_window`` prefill queries,
    max-pooled over key positions to keep contiguous clusters.

    ``Q, K : [H, S, D]``. Returns ``[H, S]``.
    """
    H, S, D = K.shape
    w = int(min(obs_window, S))
    Qo = Q[:, S - w :, :].astype(np.float64)                        # [H, w, D]
    logits = np.einsum("hwd,hjd->hwj", Qo, K.astype(np.float64)) * _scale(D)  # [H, w, S]
    # causal for window query t at absolute position (S - w + t): key j <= that.
    qpos = (S - w + np.arange(w))[:, None]                          # [w, 1]
    causal = np.arange(S)[None, :] <= qpos                          # [w, S]
    logits = np.where(causal[None], logits, -np.inf)
    p = _softmax(logits, axis=2)                                    # [H, w, S]
    imp = p.sum(axis=1)                                             # pool window -> [H, S]
    return _maxpool1d_same(imp, pool_kernel).astype(np.float32)


# --- shared block retention (fixed budget, sink+local forced) ----------------
def retain_blocks(
    importance: np.ndarray,
    block_size: int,
    n_real: int,
    budget: int,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
) -> np.ndarray:
    """Retain exactly ``budget`` blocks/head = forced sink + forced local + the
    top content blocks by ``importance``.

    Mirrors ``adakv.selector.select_blocks`` accounting so the realized budget
    matches AdaKV/Quest exactly. ``importance : [H, S]``. Returns mask ``[H, nb]``.
    The effective floor is ``sink + local + c_min`` (budget is clamped up to it),
    matching the rest of the framework.
    """
    H, S = importance.shape
    nb = (n_real + block_size - 1) // block_size
    blk = np.zeros((H, nb), dtype=np.float64)
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n_real)
        blk[:, b] = importance[:, s:e].sum(axis=1)

    sink = min(n_sink_blocks, nb)
    local = min(n_local_blocks, nb)
    biased = blk.copy()
    if sink:
        biased[:, :sink] = np.inf
    if local:
        biased[:, nb - local :] = np.inf

    floor = min(sink + local + c_min, nb)
    k = min(max(int(budget), floor), nb)
    order = np.argsort(-biased, axis=1)                 # descending
    mask = np.zeros((H, nb), dtype=bool)
    for h in range(H):
        mask[h, order[h, :k]] = True
        if sink:
            mask[h, :sink] = True
        if local:
            mask[h, nb - local :] = True
    return mask


def h2o_retained_mask(
    Q: np.ndarray,
    K: np.ndarray,
    block_size: int = 16,
    budget: int = 8,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
) -> np.ndarray:
    """Frozen H2O retained-block mask ``[H, nb]`` from the prefill (Q, K)."""
    imp = h2o_key_importance(Q, K)
    return retain_blocks(imp, block_size, K.shape[1], budget,
                         n_sink_blocks, n_local_blocks, c_min)


def snapkv_retained_mask(
    Q: np.ndarray,
    K: np.ndarray,
    block_size: int = 16,
    budget: int = 8,
    obs_window: int = 32,
    pool_kernel: int = 7,
    n_sink_blocks: int = 1,
    n_local_blocks: int = 4,
    c_min: int = 1,
) -> np.ndarray:
    """Frozen SnapKV retained-block mask ``[H, nb]`` from the prefill (Q, K)."""
    imp = snapkv_key_importance(Q, K, obs_window, pool_kernel)
    return retain_blocks(imp, block_size, K.shape[1], budget,
                         n_sink_blocks, n_local_blocks, c_min)


# --- frozen decode step ------------------------------------------------------
def frozen_sparse_attention(
    q: np.ndarray, K: np.ndarray, V: np.ndarray, retained_mask: np.ndarray, block_size: int
):
    """One decode step attending ONLY to the frozen retained blocks.

    ``q : [H, D]``, ``K, V : [H, S, D]``, ``retained_mask : [H, nb]``. The
    selection ignores ``q`` (that is the whole point -- it was fixed at prefill);
    ``q`` is used only for the attention over the retained tokens. Structurally
    identical to ``adakv.reference.sparse_attention`` masking, so a full retained
    set reproduces dense attention exactly. Returns ``(out[H, D], counts[H])``.
    """
    H, S, D = K.shape
    tok = np.repeat(retained_mask, block_size, axis=1)[:, :S]   # block mask -> token mask
    a = np.einsum("hd,hsd->hs", q, K) / np.sqrt(D)
    a = np.where(tok, a, -np.inf)
    w = _softmax(a, axis=-1)
    out = np.einsum("hs,hsd->hd", w, V)
    return out, retained_mask.sum(axis=1)
