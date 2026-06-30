"""Quest baseline (in-framework) + the AdaKV budget-policy cells.

Quest = query-aware *block* sparsity with a **per-dimension min/max** page
estimator and a **fixed** per-head top-k budget (Tang et al., ICML 2024). In
AdaKV's framework this is one cell of the (estimator x budget_policy) grid, so
it runs through the *same* selection code and the *same* Triton kernel as AdaKV
-- only the scoring (minmax vs centroid) and the per-head count (fixed vs
adaptive) differ. This isolates the two algorithmic contributions and removes
the confounds of a separate implementation/kernel.

    AdaKV (entropy) : estimator="centroid", budget_policy="adaptive"
    AdaKV (nucleus) : estimator="centroid", budget_policy="adaptive_nucleus"
    Quest           : estimator="minmax",   budget_policy="fixed"

All policies pin the *mean* KV budget to avg_budget and guarantee every head
>= c_min content blocks beyond the sink/local window, so methods are compared at
an equal, audited budget and no head silently degrades to a sliding window.

Use benchmarks/quality/eval_equal_budget.py to run them at a matched budget;
realized budgets and selection recall are reported for auditability.

Reproducibility notes for the paper:
  - "Quest" here is the controlled in-framework reproduction for the
    quality-at-equal-budget comparison. Pin block_size, sink/local, c_min,
    k_min/k_max, nucleus_p in docs/design.md.
  - The latency comparison vs Quest's own CUDA kernel is a separate experiment
    (efficiency/): pin the upstream Quest commit there, since kernel
    implementation -- not just selection -- is what is being measured.
"""
from __future__ import annotations

# Config presets so callers/tests can refer to the cells by name.
ADAKV = dict(estimator="centroid", budget_policy="adaptive")
ADAKV_NUCLEUS = dict(estimator="centroid", budget_policy="adaptive_nucleus")
QUEST = dict(estimator="minmax", budget_policy="fixed")
ABLATION_ESTIMATOR_ONLY = dict(estimator="centroid", budget_policy="fixed")
ABLATION_BUDGET_ONLY = dict(estimator="minmax", budget_policy="adaptive")


def quest_config(avg_budget: int, **common) -> dict:
    """enable_adakv(**...) kwargs for the in-framework Quest cell."""
    return dict(avg_budget=avg_budget, **QUEST, **common)
