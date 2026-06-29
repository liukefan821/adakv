"""Needle-in-a-Haystack retrieval sanity check (quality at fixed KV budget).

A fast smoke test before the full RULER/LongBench sweep: plant a fact at depth d
in a context of length L, ask for it, score exact match. Sweep (L, d) and KV
budget; AdaKV should hold ~100% retrieval at budgets where eviction-based
methods (H2O/SnapKV) start missing off-window needles.

    python benchmarks/quality/run_needle.py --model <hf_id> --lengths 32000 64000 128000

STUB: wire model + AdaKV patch; emit a depth x length pass/fail heatmap.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32000, 64000, 128000])
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 30, 50, 70, 90])
    ap.add_argument("--budget", type=float, default=0.0625)
    args = ap.parse_args()
    raise NotImplementedError("Implement needle sweep; see docstring.")


if __name__ == "__main__":
    main()
