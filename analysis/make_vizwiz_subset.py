#!/usr/bin/env python3
"""
Build the fixed VizWiz-val subsample for the П2 prompt sweep.

Stratified by answerability (preserves the answerable/unanswerable ratio), fixed
seed, identical id set across all prompts and models. Reads the same gt.json used
by score_tokenf1.py, so no extra dependency.

  python3 make_vizwiz_subset.py --gt gt.json --n 1500 --seed 0 \
                                --out vizwiz_val_sub1500_seed0.txt

gt.json : { id: {"answers": [...], "answerable": true|false}, ... }
out     : one id per line (the ids your runner filters the dataset to).
"""
import argparse, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vizwiz_val_sub1500_seed0.txt")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    # VizWiz ids only (skip vqav2 if gt.json is shared)
    ids = [i for i in gt if "vizwiz" in i.lower()]
    ans = sorted(i for i in ids if gt[i]["answerable"])
    una = sorted(i for i in ids if not gt[i]["answerable"])
    total = len(ans) + len(una)
    if total == 0:
        raise SystemExit("no vizwiz ids in gt.json")

    n = min(a.n, total)
    n_ans = round(n * len(ans) / total)          # preserve the ratio
    n_una = n - n_ans
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(ans, n_ans) + rng.sample(una, n_una))

    with open(a.out, "w") as f:
        f.write("\n".join(pick) + "\n")
    print(f"pool: {len(ans)} answerable + {len(una)} unanswerable = {total}")
    print(f"wrote {len(pick)} ids ({n_ans} ans + {n_una} unans, "
          f"{100*n_ans/len(pick):.1f}% answerable) -> {a.out}")


if __name__ == "__main__":
    main()
