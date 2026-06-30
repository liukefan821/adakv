"""Equal-budget quality comparison: AdaKV vs Quest (+ ablations) on multi-key NIAH.

Phase-2 headline experiment. Runs the (estimator x budget_policy) grid through
ONE attention backend and the SAME model, so the only thing that varies across
methods is how blocks are scored and how many per head:

    method      estimator   budget_policy       what it is
    ---------   ---------   -----------------   --------------------------------
    full        --          --                  dense attention (quality ceiling)
    adakv       centroid    adaptive            ours (entropy budget)
    adakv_nuc   centroid    adaptive_nucleus    ours (nucleus / mass budget)
    quest       minmax      fixed               Quest baseline (in-framework)
    est_only    centroid    fixed               ablation: estimator only
    bud_only    minmax      adaptive            ablation: budget only

Every budget policy pins the MEAN budget to avg_budget and guarantees each head
>= c_min content blocks beyond the sink/local window (no silent collapse to a
sliding window). For each (method, budget) the script reports:
  - realized mean blocks/head  (proves the budgets are actually matched);
  - selection recall           (fraction of decode steps where the needle block
                                is kept by >=1 head -- low-variance, isolates
                                selection quality from generation noise);
  - answer accuracy            (exact-substring match of the planted code).

Task: multi-key needle-in-a-haystack. Plant K distinct "secret <city> code is
<7-digit>" facts at K evenly-spread depths in an L-token context, then ask for
each city's code. With --needles-per-query N>1, each query asks for N codes at
once: this forces a single forward pass to surface N needle blocks, creating
per-head demand > 1 and uneven across heads -- the regime where an *adaptive*
per-head budget can beat a *fixed* one (single-needle queries only need 1 block
per head, so adaptive and fixed coincide there).

Run (Colab T4):
    python benchmarks/quality/eval_equal_budget.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --ctx 6000 --needles 6 --trials 4 --budgets 4 6 8 12 \
        --out results/equal_budget_qwen1p5b.csv

Heavier sweep once the headline looks right:
    --ctx 16000 --needles 8 --trials 5 --budgets 6 8 12
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch

from adakv.patch import install_adakv, enable_adakv, disable_adakv
from adakv.attention import (
    reset_budget_trace, get_budget_trace, stop_budget_trace,
    set_recall_target, get_recall, clear_recall_target,
)

# (estimator, budget_policy) for each non-dense method.
METHODS = {
    "adakv":     ("centroid", "adaptive"),
    "adakv_nuc": ("centroid", "adaptive_nucleus"),
    "quest":     ("minmax",   "fixed"),
    "est_only":  ("centroid", "fixed"),
    "bud_only":  ("minmax",   "adaptive"),
}

CITIES = ["Lisbon", "Nairobi", "Sapporo", "Calgary", "Brisbane",
          "Helsinki", "Medellin", "Chengdu", "Tbilisi", "Reykjavik"]

FILLER = ("The history of cartography is long and varied; maps have guided "
          "travelers across deserts and oceans for thousands of years. ")


def build_multikey_prompt(tok, n_tokens, cities, codes):
    """Plant len(cities) facts at evenly-spread depths in an n_tokens context."""
    reps = n_tokens // max(len(tok(FILLER).input_ids), 1) + 2
    base = tok(FILLER * reps).input_ids[:n_tokens]
    K = len(cities)
    depths = [(i + 1) / (K + 1) for i in range(K)]
    items = sorted(zip(depths, cities, codes), key=lambda t: t[0], reverse=True)
    ids = list(base)
    for d, city, code in items:
        fact = tok(f"\n\nImportant: the secret {city} code is {code}.\n\n",
                   add_special_tokens=False).input_ids
        ids[int(len(base) * d):int(len(base) * d)] = fact
    return tok.decode(ids)


def make_dataset(tok, n_tokens, n_needles, n_trials, seed):
    rng = random.Random(seed)
    data = []
    for _ in range(n_trials):
        cities = rng.sample(CITIES, n_needles)
        codes = [f"{rng.randint(1000000, 9999999)}" for _ in cities]
        prompt = build_multikey_prompt(tok, n_tokens, cities, codes)
        data.append((prompt, cities, codes))
    return data


def find_block_ids(input_ids, sub_ids, block_size):
    """Block indices in input_ids covered by the contiguous subsequence sub_ids."""
    n, m = len(input_ids), len(sub_ids)
    for i in range(n - m + 1):
        if input_ids[i:i + m] == sub_ids:
            return sorted({(i + j) // block_size for j in range(m)})
    return []


@torch.no_grad()
def answer(model, tok, context, pairs, block_size, max_new_tokens, do_recall):
    """Ask for the codes of all (city, code) in `pairs` in ONE query.

    Asking for several needles at once forces a single forward pass to surface
    multiple needle blocks, creating per-head demand > 1 (and uneven across
    heads) -- the regime where an adaptive per-head budget can beat a fixed one.

    Returns (n_correct, n_asked, mean_recall, ctx_len).
    """
    if len(pairs) == 1:
        city = pairs[0][0]
        q = (f"{context}\n\nQuestion: What is the secret {city} code? "
             f"Reply with only the 7-digit number.")
    else:
        names = ", ".join(c for c, _ in pairs)
        q = (f"{context}\n\nQuestion: What are the secret codes for {names}? "
             f"Reply with one line per city in the form 'City: code'.")
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    ids = inputs["input_ids"][0].tolist()
    ctx_len = len(ids)
    if do_recall:
        blocks = set()
        for _, code in pairs:
            b = find_block_ids(ids, tok(f" {code}", add_special_tokens=False).input_ids, block_size)
            if not b:
                b = find_block_ids(ids, tok(code, add_special_tokens=False).input_ids, block_size)
            blocks.update(b)
        set_recall_target(sorted(blocks))
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0, ctx_len:], skip_special_tokens=True)
    rec = get_recall() if do_recall else float("nan")
    if do_recall:
        clear_recall_target()
    n_correct = sum(int(code in text) for _, code in pairs)
    return n_correct, len(pairs), rec, ctx_len


def make_queries(cities, codes, npq, rng):
    """Group the planted needles into non-overlapping queries of size npq."""
    idx = list(range(len(cities)))
    rng.shuffle(idx)
    groups = [idx[i:i + npq] for i in range(0, len(idx), npq)]
    groups = [g for g in groups if len(g) == npq] or [idx[:npq]]
    return [[(cities[j], codes[j]) for j in g] for g in groups]


def eval_method(model, tok, data, method, budget, common, block_size, max_new_tokens,
                npq, seed):
    """Return (accuracy, realized_mean_blocks, mean_recall, ctx_len)."""
    is_sparse = method != "full"
    if is_sparse:
        est, pol = METHODS[method]
        enable_adakv(avg_budget=budget, estimator=est, budget_policy=pol,
                     block_size=block_size, **common)
        reset_budget_trace()
    else:
        disable_adakv()

    rng = random.Random(seed)
    correct = total = 0
    recalls = []
    ctx_len = 0
    # bump max_new_tokens for multi-needle answers (need room for N lines)
    mnt = max_new_tokens if npq == 1 else max(max_new_tokens, 24 * npq)
    for prompt, cities, codes in data:
        for pairs in make_queries(cities, codes, npq, rng):
            nc, na, rec, ctx_len = answer(model, tok, prompt, pairs,
                                          block_size, mnt, is_sparse)
            correct += nc; total += na
            if is_sparse and rec == rec:  # not nan
                recalls.append(rec)
    if is_sparse:
        realized = float(np.mean(get_budget_trace())); stop_budget_trace()
        recall = float(np.mean(recalls)) if recalls else float("nan")
    else:
        realized = recall = float("nan")
    return correct / max(total, 1), realized, recall, ctx_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=6000)
    ap.add_argument("--needles", type=int, default=6)
    ap.add_argument("--needles-per-query", type=int, default=1,
                    help="how many codes to ask for in a single query; >1 stresses "
                         "per-head budget (the regime where adaptive can beat fixed)")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--methods", nargs="+",
                    default=["full", "quest", "adakv", "adakv_nuc", "est_only", "bud_only"])
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=512)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--n-local", type=int, default=2)
    ap.add_argument("--c-min", type=int, default=1)
    ap.add_argument("--nucleus-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/equal_budget.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=dev)
    install_adakv(model)
    print(f"loaded | device {model.device} | attn {model.config._attn_implementation}")

    data = make_dataset(tok, args.ctx, args.needles, args.trials, args.seed)
    npq = max(1, args.needles_per_query)
    qper = (args.needles // npq) * npq          # codes asked per trial
    n_codes = qper * args.trials                 # total code-retrievals scored/cell
    common = dict(k_min=args.k_min, k_max=args.k_max, n_sink_blocks=args.n_sink,
                  n_local_blocks=args.n_local, c_min=args.c_min,
                  nucleus_p=args.nucleus_p, temperature=args.temperature)

    rows = []
    print(f"\nmulti-key NIAH | ctx≈{args.ctx} tok | {args.needles} needles x "
          f"{args.trials} trials | {npq} needle(s)/query = {n_codes} code-retrievals/cell "
          f"| sink {args.n_sink} local {args.n_local} c_min {args.c_min}\n")
    header = f"{'method':<10} {'target':>6} {'realized':>9} {'recall':>7} {'acc':>7}"
    print(header); print("-" * len(header))

    if "full" in args.methods:
        acc, real, rec, ctx = eval_method(model, tok, data, "full", 0, common,
                                           args.block_size, args.max_new_tokens, npq, args.seed)
        print(f"{'full':<10} {'--':>6} {'--':>9} {'--':>7} {acc:>7.3f}")
        rows.append(dict(method="full", target_budget="", realized_budget="",
                         recall="", accuracy=acc, ctx_tokens=ctx, queries=n_codes))

    for b in args.budgets:
        for m in args.methods:
            if m == "full":
                continue
            acc, real, rec, ctx = eval_method(model, tok, data, m, b, common,
                                              args.block_size, args.max_new_tokens, npq, args.seed)
            print(f"{m:<10} {b:>6} {real:>9.3f} {rec:>7.3f} {acc:>7.3f}")
            rows.append(dict(method=m, target_budget=b, realized_budget=round(real, 3),
                             recall=round(rec, 3), accuracy=acc, ctx_tokens=ctx, queries=n_codes))
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "target_budget", "realized_budget",
                                          "recall", "accuracy", "ctx_tokens", "queries"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")
    print("\nHeadline: adakv / adakv_nuc vs quest at matched realized budget. "
          "Read 'recall' first (low variance): does selection keep the needle? "
          "Then 'acc'. est_only / bud_only isolate the two contributions.")


if __name__ == "__main__":
    main()
