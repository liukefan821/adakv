"""Quest baseline (in-framework) — the headline comparison for AdaKV.

Quest = query-aware *block* sparsity with a **per-dimension min/max** page
estimator and a **fixed** per-head top-k budget (Tang et al., ICML 2024). In
AdaKV's framework this is exactly one cell of the (estimator x budget_policy)
grid, so we run it through the *same* selection code and the *same* Triton
kernel as AdaKV — only the scoring (minmax vs centroid) and the per-head count
(fixed vs adaptive) differ. This isolates the two algorithmic contributions and
removes the confounds of comparing against a separate implementation/kernel.

    AdaKV : enable_adakv(estimator="centroid", budget_policy="adaptive", avg_budget=B)
    Quest : enable_adakv(estimator="minmax",   budget_policy="fixed",    avg_budget=B)

Use benchmarks/quality/eval_equal_budget.py to run both at a matched KV budget
and compare retrieval accuracy; both realized budgets are reported so the
comparison is auditable.

Reproducibility notes for the paper:
  - "Quest" here is the *controlled in-framework* reproduction used for the
    quality-at-equal-budget comparison. Pin the exact settings (block_size,
    sink/local, k_min/k_max) in docs/design.md.
  - The *latency* comparison against Quest's own CUDA kernel is a separate
    experiment (efficiency/): for that, pin the upstream Quest commit, since
    kernel implementation — not just selection — is what is being measured.
"""
from __future__ import annotations

# Config presets so callers/tests can refer to the cells by name.
ADAKV = dict(estimator="centroid", budget_policy="adaptive")
QUEST = dict(estimator="minmax", budget_policy="fixed")
ABLATION_ESTIMATOR_ONLY = dict(estimator="centroid", budget_policy="fixed")
ABLATION_BUDGET_ONLY = dict(estimator="minmax", budget_policy="adaptive")


def quest_config(avg_budget: int, **common) -> dict:
    """Return an enable_adakv(**...) kwargs dict for the in-framework Quest cell."""
    return dict(avg_budget=avg_budget, **QUEST, **common)
