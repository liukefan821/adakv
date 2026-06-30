"""Equal-budget quality comparison: AdaKV vs Quest (+ ablation cells) on multi-key NIAH.

This is the Phase-2 headline experiment. It runs the (estimator x budget_policy)
grid through *one* attention backend and the *same* model, so the only thing that
varies across methods is how blocks are scored and how many per head:

    method     estimator   budget_policy   what it is
    --------   ---------   -------------   ---------------------------------------
    full       --          --              dense attention (quality ceiling)
    adakv      centroid    adaptive        ours
    quest      minmax      fixed           Quest baseline (in-framework, same kernel)
    est_only   centroid    fixed           ablation: estimator contribution only
    bud_only   minmax      adaptive        ablation: budget contribution only

Task: multi-key needle-in-a-haystack. Plant K distinct "secret <city> code is
<7-digit>" facts at K evenly-spread depths in an L-token context, then ask for
each city's code separately and score exact-substring match. Multi-key is the
discriminative version (single-key saturates for AdaKV already): at a low KV
budget, the looser min/max estimator + fixed budget should start dropping needles
where the centroid estimator + adaptive budget hold.

For every (method, budget) the script reports retrieval accuracy AND the realized
mean blocks/head measured on the actual decode trace, so the equal-budget claim is
auditable (target budgets and realized budgets are both printed).

Run (Colab T4):
    python benchmarks/quality/eval_equal_budget.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --ctx 4000 --needles 6 --budgets 6 8 12 16 \
        --out results/equal_budget_qwen1p5b.csv

Heavier sweep once the headline looks right:
    --ctx 16000 --needles 8 --budgets 8 16 32
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch

from adakv.patch import install_adakv, enable_adakv, disable_adakv
from adakv.attention import reset_budget_trace, get_budget_trace, stop_budget_trace

# (estimator, budget_policy) for each non-dense method.
METHODS = {
    "adakv":    ("centroid", "adaptive"),
    "quest":    ("minmax",   "fixed"),
    "est_only": ("centroid", "fixed"),
    "bud_only": ("minmax",   "adaptive"),
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
    # depths spread in (0,1), avoiding the very edges
    depths = [(i + 1) / (K + 1) for i in range(K)]
    # insert from the back so earlier insert positions stay valid
    items = sorted(zip(depths, cities, codes), key=lambda t: t[0], reverse=True)
    ids = list(base)
    for d, city, code in items:
        fact = tok(f"\n\nImportant: the secret {city} code is {code}.\n\n",
                   add_special_tokens=False).input_ids
        pos = int(len(base) * d)
        ids[pos:pos] = fact
    return tok.decode(ids)


def make_dataset(tok, n_tokens, n_needles, n_trials, seed):
    """n_trials independent contexts, each with n_needles (city, code) facts."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_trials):
        cities = rng.sample(CITIES, n_needles)
        codes = [f"{rng.randint(1000000, 9999999)}" for _ in cities]
        prompt = build_multikey_prompt(tok, n_tokens, cities, codes)
        data.append((prompt, cities, codes))
    return data


@torch.no_grad()
def answer(model, tok, context, city, max_new_tokens):
    q = (f"{context}\n\nQuestion: What is the secret {city} code? "
         f"Reply with only the 7-digit number.")
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    ctx_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0, ctx_len:], skip_special_tokens=True)
    return text, ctx_len


def eval_method(model, tok, data, method, budget, common, max_new_tokens):
    """Return (accuracy, realized_mean_blocks_per_head, ctx_len_example)."""
    if method == "full":
        disable_adakv()
    else:
        est, pol = METHODS[method]
        enable_adakv(avg_budget=budget, estimator=est, budget_policy=pol, **common)

    if method != "full":
        reset_budget_trace()
    correct = total = 0
    ctx_example = 0
    for prompt, cities, codes in data:
        for city, code in zip(cities, codes):
            text, ctx_example = answer(model, tok, prompt, city, max_new_tokens)
            correct += int(code in text)
            total += 1
    realized = float(np.mean(get_budget_trace())) if method != "full" else float("nan")
    if method != "full":
        stop_budget_trace()
    return correct / max(total, 1), realized, ctx_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=4000, help="context length in tokens")
    ap.add_argument("--needles", type=int, default=6, help="distinct facts per context")
    ap.add_argument("--trials", type=int, default=3, help="independent contexts")
    ap.add_argument("--budgets", type=int, nargs="+", default=[6, 8, 12, 16],
                    help="avg blocks/head target (mean KV budget)")
    ap.add_argument("--methods", nargs="+",
                    default=["full", "quest", "adakv", "est_only", "bud_only"])
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=256)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--n-local", type=int, default=4)
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
    n_q = args.needles * args.trials
    common = dict(block_size=args.block_size, k_min=args.k_min, k_max=args.k_max,
                  n_sink_blocks=args.n_sink, n_local_blocks=args.n_local,
                  temperature=args.temperature)

    rows = []
    print(f"\nmulti-key NIAH | ctx≈{args.ctx} tok | {args.needles} needles x "
          f"{args.trials} trials = {n_q} queries/cell\n")
    header = f"{'method':<9} {'target':>6} {'realized':>9} {'acc':>7}   {'queries':>7}"
    print(header); print("-" * len(header))

    # full once (budget-independent); sparse cells for each budget.
    if "full" in args.methods:
        acc, real, ctx = eval_method(model, tok, data, "full", 0, common, args.max_new_tokens)
        print(f"{'full':<9} {'--':>6} {'--':>9} {acc:>7.3f}   {n_q:>7}")
        rows.append(dict(method="full", target_budget="", realized_budget="",
                         accuracy=acc, ctx_tokens=ctx, queries=n_q))

    for b in args.budgets:
        for m in args.methods:
            if m == "full":
                continue
            acc, real, ctx = eval_method(model, tok, data, m, b, common, args.max_new_tokens)
            print(f"{m:<9} {b:>6} {real:>9.3f} {acc:>7.3f}   {n_q:>7}")
            rows.append(dict(method=m, target_budget=b, realized_budget=round(real, 3),
                             accuracy=acc, ctx_tokens=ctx, queries=n_q))
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "target_budget", "realized_budget",
                                          "accuracy", "ctx_tokens", "queries"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")
    print("\nHeadline: compare adakv vs quest at matched 'realized' budget. "
          "est_only / bud_only isolate the two contributions.")


if __name__ == "__main__":
    main()
