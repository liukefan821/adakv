"""Plug AdaKV into a HuggingFace model via the attention interface (transformers >= 5.x).

Usage:
    from adakv.patch import install_adakv, enable_adakv, disable_adakv
    install_adakv(model)
    enable_adakv(block_size=16, avg_budget=16)
    # ... model.generate(...) ...
    disable_adakv()

Only single-sequence (batch 1), single-token decode steps are intercepted; prefill
and batched calls delegate to sdpa (dense). Centroids are recomputed from the full
KV each decode step (correctness-first; incremental update is a separate optimisation).
Validated: needle retrieved at ~1/30 KV budget on Qwen2.5-1.5B-Instruct.
"""
from __future__ import annotations

import torch  # noqa: F401

from .attention import AdaKVCache, plan_selection
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
                  c_min=1, nucleus_p=0.9)
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]


def adakv_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    cfg = _ADAKV_CFG
    B, Hq, q_len, _ = query.shape
    if (not cfg["enabled"]) or q_len != 1 or B != 1:
        return _SDPA(module, query, key, value, attention_mask=attention_mask,
                     scaling=scaling, dropout=dropout, **kwargs)
    k = key[0].contiguous()
    v = value[0].contiguous()
    qh = query[0, :, 0, :].contiguous()
    cache = AdaKVCache(block_size=cfg["block_size"], estimator=cfg["estimator"]).append_prefill(k, v)
    bt, sl = plan_selection(qh, cache, cfg["avg_budget"], cfg["k_min"], cfg["k_max"],
                            cfg["n_sink_blocks"], cfg["n_local_blocks"], cfg["temperature"],
                            estimator=cfg["estimator"], budget_policy=cfg["budget_policy"],
                            c_min=cfg["c_min"], nucleus_p=cfg["nucleus_p"])
    if HAS_TRITON and qh.is_cuda:
        out = block_sparse_decode(qh, k, v, bt, sl, cfg["block_size"], sm_scale=scaling)
    else:
        out = _torch_masked_decode(qh, k, v, bt, sl, cfg["block_size"], cache.centroid.shape[1])
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


def disable_adakv():
    _ADAKV_CFG["enabled"] = False
