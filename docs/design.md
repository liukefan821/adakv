# AdaKV — Design & Positioning

Query-aware, **per-head-adaptive** block-sparse attention for the **decode** phase
of long-context LLM inference, with a hand-written fused block-sparse kernel.

## Problem

At long context, decode is memory-bound: each generated token attends over the
entire KV cache, so latency scales with sequence length and is dominated by KV
memory traffic, not FLOPs. Query-aware sparsity (attend to only the blocks that
matter for the current query) cuts that traffic — but existing methods leave two
things on the table.

## Contribution (the novelty triple)

1. **Sharper, cheaper block estimator.** Summarise each KV block by its *centroid*
   (mean key); decode-time block score is one dot product `q·centroid`. Cheaper
   than Quest's per-dimension min/max bound (1× vs 2× summary memory) and, because
   it estimates the block's *typical* alignment rather than a loose upper bound,
   empirically a better *ranking* signal at equal budget. (`minmax` is retained for
   ablation.) — `estimator.py`
2. **Adaptive per-head budget.** A global KV budget is split across heads by the
   *concentration* of each head's block-score distribution (normalised entropy):
   sharply-peaked "local" heads get few blocks, flat "retrieval" heads get many.
   Training-free, O(H·n_blocks)/step. Prior query-aware methods use a **fixed**
   top-k per head/layer. — `budget.py`
3. **Fused ragged block-sparse decode kernel.** A FlashAttention-style decode
   kernel that takes a *variable-length* per-head block list and gathers only those
   KV blocks inside the kernel — no host-side KV compaction. Handling the ragged
   per-head count in one launch is the part FlashAttention/SDPA cannot do. —
   `kernels/block_sparse_decode.py`

Stabilisers: attention **sinks** (forced leading blocks) + **sliding window**
(forced trailing blocks), per StreamingLLM. — `selector.py`

## Differentiation

| Method | Granularity | Per-query dynamic? | Permanent eviction? | Budget | Phase | vs AdaKV |
|---|---|---|---|---|---|---|
| StreamingLLM | token | no (fixed sink+window) | n/a (window) | fixed | decode | no content-based selection; misses off-window facts |
| H2O | token | accumulated, not per-query | **yes** (lossy) | fixed | decode | evicts → fails on later-needed tokens; AdaKV keeps full cache |
| SnapKV | token | prompt-time only | **yes** (compress prompt) | fixed | prefill→decode | one-shot compression; AdaKV re-selects every step |
| Quest | **block** | **yes** | no | **fixed top-k** | decode | min/max estimator + fixed budget; AdaKV = centroid + adaptive budget |
| MInference | block/pattern | offline head patterns | no | pattern | **prefill** | targets prefill; AdaKV targets decode, online |
| **AdaKV** | **block** | **yes** | **no** | **adaptive per-head** | **decode** | — |

**Clean separation from prior KV-cache work in this portfolio:** AdaKV is a pure
*efficiency* method — it keeps the full cache and selects dynamically, with no
verifiability/cryptographic component. It does **not** overlap with verifiable-
inference / commitment work (different problem, different machinery). State this
explicitly in the paper's related-work to pre-empt any salami-slicing concern.

## Benchmark protocol (what makes the paper stand)

**Models.** Llama-3.1-8B-Instruct (128K) primary; Qwen2.5-7B-Instruct secondary.

**Quality at fixed KV budget** — task score vs budget ∈ {1/4, 1/8, 1/16} of full KV:
- RULER (4K–128K) — primary, synthetic multi-task long-context
- LongBench / LongBench-v2 — realistic tasks
- Needle-in-a-Haystack — retrieval sanity (depth × length heatmap)
- Baselines: Full attention (upper bound), StreamingLLM, H2O, SnapKV, **Quest** (headline)
- **Target:** match full attention within ≈1 point at a budget where Quest drops
  more; ≥ Quest at equal budget. Show the adaptive budget helps via ablation
  (fixed-k vs adaptive at equal mean budget).

**Efficiency** — decode latency/throughput vs context ∈ {32K, 64K, 128K}:
- AdaKV kernel vs FlashAttention-2/3 dense decode, and vs Quest's kernel at equal budget
- KV bytes touched / memory saved
- **Target:** speedup grows with context (attend to O(budget), not O(seq)); at 128K,
  decode meaningfully faster than dense; kernel ≥ Quest at equal budget from fusion.

**Hardware.** Report on ≥2 GPUs (e.g. A100 + H100). Rent by the hour for final figures.

**Honesty.** Sparse attention loses at short context (selection not amortised) and
the win is in the long-context, low-batch *decode* regime — identify and plot the
crossover; do not claim universal speedup.

## Ablations (must-have)

- estimator: centroid vs minmax vs random-projection
- budget: adaptive vs fixed-k (equal mean budget)
- stabilisers: with/without sink, with/without sliding window
- block size: 8 / 16 / 32 / 64 (quality–latency tradeoff)
- sensitivity to `avg_budget`, `k_min`, `k_max`, `temperature`
