"""One-line drop-in patching for HuggingFace models.

Usage:
    from adakv.patch import patch_model
    model = AutoModelForCausalLM.from_pretrained(name, ...)
    patch_model(model, block_size=16, avg_budget=8)   # decode now uses AdaKV

STUB: wire ``adakv_decode_attention`` into the target model's attention forward.
Start with Llama-3.1 (``LlamaAttention``) and Qwen2.5 (``Qwen2Attention``); both
expose past_key_values you can route through ``AdaKVCache``. Keep prefill dense
and only intercept the decode step (q_len == 1) so correctness is easy to bisect.
"""
from __future__ import annotations

SUPPORTED = {
    "LlamaAttention": "llama",
    "Qwen2Attention": "qwen2",
}


def patch_model(model, block_size: int = 16, avg_budget: int = 8, **kwargs):
    """Replace attention forward on supported decoder layers.

    TODO:
      1. locate attention submodules by class name (see SUPPORTED);
      2. attach an AdaKVCache per layer/head-group;
      3. on decode (q_len == 1) call adakv.attention.adakv_decode_attention;
      4. fall back to the original forward during prefill.
    """
    patched = 0
    for module in model.modules():
        if type(module).__name__ in SUPPORTED:
            # _install_adakv_forward(module, block_size, avg_budget, **kwargs)
            patched += 1
    if patched == 0:
        raise ValueError(
            f"No supported attention modules found. Supported: {list(SUPPORTED)}"
        )
    return model
