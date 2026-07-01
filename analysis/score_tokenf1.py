#!/usr/bin/env python3
"""
A1 / П2 — third, defensible matcher: normalised token-F1.

Answers the self-review's Major 3.2: show the model ranking moves between two
*reasonable* matchers (strict exact-match and token-F1), not only between strict
and the lenient upper bound. Re-scores the saved per-prediction outputs against the
reference answers; no GPU, runs in seconds on the machine that has the dataset.

INPUTS
  --items DIR     directory of *.items.jsonl (id, pred, acc, abstain), one per unit
                  (only the clean VQAv2 and clean VizWiz units are used here)
  --gt    FILE    json: { id: {"answers": [10 reference strings],
                              "answerable": true|false} }
                  Your harness already has this; dump it once.
  --bracket FILE  a1_bracket.json (for strict/lenient, to print the full tau matrix)
  --out   FILE    where to write the per-model token-F1 json

SCOPE (mirrors the bracket in the paper)
  VQAv2          : all items
  VizWiz (ans.)  : answerable items only

OUTPUT
  per-model token-F1 on VQAv2 and VizWiz-answerable, plus the Kendall-tau matrix
  among {strict, lenient, token-F1} on each dataset.
"""
import argparse, glob, json, os, re, string
from collections import Counter
import numpy as np
import scipy.stats as st

MODELS = ["qwen3vl_8b", "qwen25vl_7b", "llava16_13b", "llava16_7b",
          "internvl3_8b", "internvl35_8b", "pixtral_12b", "idefics3_8b", "gemma3_12b"]
DISP = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen25vl_7b": "Qwen2.5-VL-7B",
        "llava16_13b": "LLaVA-1.6-13B", "llava16_7b": "LLaVA-1.6-7B",
        "internvl3_8b": "InternVL3-8B", "internvl35_8b": "InternVL3.5-8B",
        "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B",
        "gemma3_12b": "Gemma-3-12B"}

_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ART.sub(" ", s)
    return s.split()


def f1(pred, ref):
    p, r = norm(pred), norm(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def score_item(pred, refs):
    return max(f1(pred, r) for r in refs)


def load_items(d, model, dataset):
    f = os.path.join(d, f"a1__{model}__{dataset}__clean__seed0.items.jsonl")
    return [json.loads(l) for l in open(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--bracket", default=None)
    ap.add_argument("--out", default="a1_tokenf1.json")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    res = {}
    for m in MODELS:
        row = {}
        for ds, key in [("vqav2_val", "vqav2"), ("vizwiz_val", "vizwiz_ans")]:
            items = load_items(a.items, m, ds)
            vals = []
            for it in items:
                g = gt.get(it["id"])
                if g is None:
                    continue
                if ds == "vizwiz_val" and not g["answerable"]:
                    continue            # VizWiz: answerable subset only
                vals.append(score_item(it["pred"], g["answers"]))
            row[key + "_f1"] = float(np.mean(vals)) if vals else float("nan")
            row[key + "_n"] = len(vals)
        res[m] = row

    print(f"{'model':14s} {'VQAv2 F1':>9s} {'VZ-ans F1':>10s}")
    for m in MODELS:
        print(f"{DISP[m]:14s} {res[m]['vqav2_f1']:9.3f} {res[m]['vizwiz_ans_f1']:10.3f}")

    if a.bracket and os.path.isfile(a.bracket):
        brk = json.load(open(a.bracket))
        for ds, sk, lk, fk in [("VQAv2", "vqav2_strict", "vqav2_lenient", "vqav2_f1"),
                               ("VizWiz-ans", "vizwiz_ans_strict", "vizwiz_ans_lenient", "vizwiz_ans_f1")]:
            strict = [brk[m][sk] for m in MODELS]
            lenient = [brk[m][lk] for m in MODELS]
            tf1 = [res[m][fk] for m in MODELS]
            print(f"\n[{ds}] Kendall tau:")
            for name, x, y in [("strict vs lenient", strict, lenient),
                               ("strict vs F1     ", strict, tf1),
                               ("lenient vs F1    ", lenient, tf1)]:
                t, p = st.kendalltau(x, y)
                print(f"   {name}: tau={t:.3f}  p={p:.3f}")

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
