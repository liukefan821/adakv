"""Decode latency / throughput vs context length.

Sparse attention only pays off once the KV cache is long enough to amortise the
selection overhead, so this sweeps context length and reports the crossover.

    python benchmarks/efficiency/bench_decode_latency.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --ctx 8192 32768 65536 131072 --budget 0.0625

Reports, per context length: dense (FlashAttention/SDPA) vs AdaKV decode
latency (ms/token), tokens/s, and KV bytes touched. Use torch.cuda.Event for
timing with warmup; report median over >=50 steps.

STUB: fill in model loading + the two timed paths. Keep dense and AdaKV reading
the *same* cache so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 65536, 131072])
    ap.add_argument("--budget", type=float, default=0.0625, help="fraction of full KV")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    raise NotImplementedError(
        f"Implement timed dense-vs-AdaKV decode for ctx={args.ctx}, "
        f"budget={args.budget}. See module docstring."
    )


if __name__ == "__main__":
    main()
