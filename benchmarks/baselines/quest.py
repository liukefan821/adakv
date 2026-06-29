"""Quest baseline adapter (fixed per-head top-k, min/max page estimator).

The headline comparison for AdaKV. Run Quest and AdaKV at *equal* average KV
budget and compare both quality (RULER/LongBench) and decode latency. AdaKV's
expected wins: (1) better quality at equal budget from the centroid estimator +
adaptive per-head budget; (2) competitive-or-faster kernel from fused gather.

Reference: Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context
LLM Inference", ICML 2024. Pin the exact commit/version you compare against and
record it in docs/design.md so the comparison is reproducible.

STUB: wrap the upstream Quest kernel (or reimplement its selection) behind the
same interface as adakv.attention.adakv_decode_attention.
"""
raise NotImplementedError("Pin upstream Quest and adapt to the common interface.")
