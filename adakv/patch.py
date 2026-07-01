"""Plug AdaKV into a HuggingFace model via the attention interface (transformers >= 5.x).

Usage:
    from adakv.patch import install_adakv, enable_adakv, disable_adakv
    install_adakv(model)
    enable_adakv(block_size=16, avg_budget=16)              # AdaKV (default)
    enable_adakv(block_size=16, avg_budget=16, selection="h2o")   # H2O baseline
    # ... model.generate(...) ...
    disable_adakv()

Only single-sequence (batch 1), single-token decode steps are intercepted; prefill
and batched calls delegate to sdpa (dense).

Selection modes (``cfg["selection"]``):
  - "adakv"  : query-aware, per-decode block selection over the FULL cache
               (centroid/minmax estimator x fixed/adaptive budget). Recomputed
               every step; no permanent eviction.
  - "h2o"    : eviction baseline. On the prefill pass we capture this layer's
               (Q, K) and freeze the retained block set by ACCUMULATED causal
               attention (heavy hitters); every decode step reuses that frozen
               set (lossy, query-agnostic).
  - "snapkv" : eviction baseline. Same, but the retained set is scored by the
               last ``obs_window`` prompt queries + key-axis max-pool.
The frozen retention math lives in ``adakv.baselines_rt`` (CPU-parity-tested
against the NumPy oracle). All modes select ``budget`` blocks/head and run the
SAME kernel, so an equal-budget comparison isolates evict-vs-re-select.
"""
from __future__ import annotations

import torch  # noqa: F401

from . import baselines_rt as _brt
from .attention import AdaKVCache, plan_selection, record_selection
from .kernels.block_sparse_decode import HAS_TRITON, block_sparse_decode
from .runtime import _torch_masked_decode

try:
    from transformers import AttentionInterface
except Exception:
    from transformers.modeling_utils import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_ADAKV_CFG = dict(enabled=True, block_size=16, avg_budget=16, k_min=2, k_max=64,
                  n_sink_blocks=1, n_local_blocks=4, temperature=1.0,
                  estimator="centroid", budget_policy="adaptive",
                  c_min=1, nucleus_p=0.9,
                  # eviction-baseline knobs (only used when selection in {h2o, snapkv})
                  selection="adakv", obs_window=32, pool_kernel=7, h2o_chunk=256)
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]

# Per-layer frozen retained mask for the eviction baselines, keyed by id(module).
# Refreshed on every prefill pass and cleared by enable_adakv; within one
# generate() call every layer's prefill runs before any decode, so lookups are safe.
_FROZEN: dict[int, torch.Tensor] = {}


def _capture_frozen(module, query, key, cfg, sel):
    """Freeze this layer's retained block set from the prompt (Q, K). Called on the
    single-sequence prefill pass; does not alter the (dense) prefill output."""
    Q = query[0]                                            # [Hq, L, D]
    group = Q.shape[0] // key.shape[1]
    Kqh = _brt.expand_kv_heads(key[0].contiguous(), group)  # [Hq, L, D]
    if sel == "h2o":
        fmask = _brt.h2o_retained_mask_torch(
            Q, Kqh, block_size=cfg["block_size"], budget=cfg["avg_budget"],
            n_sink_blocks=cfg["n_sink_blocks"], n_local_blocks=cfg["n_local_blocks"],
            c_min=cfg["c_min"], chunk=cfg["h2o_chunk"])
    else:  # snapkv
        fmask = _brt.snapkv_retained_mask_torch(
            Q, Kqh, block_size=cfg["block_size"], budget=cfg["avg_budget"],
            obs_window=cfg["obs_window"], pool_kernel=cfg["pool_kernel"],
            n_sink_blocks=cfg["n_sink_blocks"], n_local_blocks=cfg["n_local_blocks"],
            c_min=cfg["c_min"])
    _FROZEN[id(module)] = fmask


def _frozen_decode_plan(qh, k, module, cfg):
    """(block_table, sel_lens) for one decode step from this layer's frozen mask,
    tracking the current tail. Records realized budget + recall like plan_selection."""
    nb_now = (k.shape[1] + cfg["block_size"] - 1) // cfg["block_size"]
    fmask = _FROZEN.get(id(module))
    if fmask is None:                                       # prefill not seen -> attend all
        fmask = torch.ones(qh.shape[0], nb_now, dtype=torch.bool, device=qh.device)
    bt, sl = _brt.frozen_decode_block_table(
        fmask, nb_now, cfg["n_sink_blocks"], cfg["n_local_blocks"])
    record_selection(bt, sl, nb_now)
    return bt, sl


def adakv_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    cfg = _ADAKV_CFG
    B, Hq, q_len, _ = query.shape
    sel = cfg.get("selection", "adakv")

    # --- prefill / batched / disabled: dense via sdpa ---
    if (not cfg["enabled"]) or q_len != 1 or B != 1:
        if cfg["enabled"] and B == 1 and q_len > 1 and sel in ("h2o", "snapkv"):
            _capture_frozen(module, query, key, cfg, sel)   # freeze retained set, still return dense
        return _SDPA(module, query, key, value, attention_mask=attention_mask,
                     scaling=scaling, dropout=dropout, **kwargs)

    # --- decode: single sequence, single token ---
    k = key[0].contiguous()
    v = value[0].contiguous()
    qh = query[0, :, 0, :].contiguous()

    if sel in ("h2o", "snapkv"):
        bt, sl = _frozen_decode_plan(qh, k, module, cfg)
    else:
        cache = AdaKVCache(block_size=cfg["block_size"], estimator=cfg["estimator"]).append_prefill(k, v)
        bt, sl = plan_selection(qh, cache, cfg["avg_budget"], cfg["k_min"], cfg["k_max"],
                                cfg["n_sink_blocks"], cfg["n_local_blocks"], cfg["temperature"],
                                estimator=cfg["estimator"], budget_policy=cfg["budget_policy"],
                                c_min=cfg["c_min"], nucleus_p=cfg["nucleus_p"])

    nb_now = (k.shape[1] + cfg["block_size"] - 1) // cfg["block_size"]
    if HAS_TRITON and qh.is_cuda:
        out = block_sparse_decode(qh, k, v, bt, sl, cfg["block_size"], sm_scale=scaling)
    else:
        out = _torch_masked_decode(qh, k, v, bt, sl, cfg["block_size"], nb_now)
    return out.unsqueeze(0).unsqueeze(0).to(query.dtype), None


AttentionInterface.register("adakv", adakv_attention_forward)


def install_adakv(model):
    try:
        model.set_attn_implementation("adakv")
    except Exception:
        model.config._attn_implementation = "adakv"
    return model


def enable_adakv(**cfg):
    _ADAKV_CFG.update(cfg)
    _ADAKV_CFG["enabled"] = True
    _FROZEN.clear()               # drop any frozen masks from a previous config


def disable_adakv():
    _ADAKV_CFG["enabled"] = False
