#!/usr/bin/env python3
"""
A1 / П2 — prompt-sensitivity comparison.

Answers the self-review's Major 3.1: is the abstention behaviour that drives the
aggregate a property of the models or of the fixed "reply unanswerable" prompt?
Run your existing split-stats script once per prompt variant, then point this at
the resulting JSONs. It tabulates, per model and per prompt, the abstention rate,
answering ability (answerable subset), correct-abstention (unanswerable subset) and
the aggregate, and reports how the abstention rate and the aggregate RANKING move
across prompts.

USAGE
  python3 compare_prompts.py P0=a1_split_P0.json P1=a1_split_P1.json \
                             P2=a1_split_P2.json P3=a1_split_P3.json

Each file is a split-stats json in the SAME schema as a1_split_stats.json
(keys per model: abstain_rate_clean, vizwiz_clean_answerable,
vizwiz_clean_unanswerable, vizwiz_clean_all).
"""
import sys, json
import numpy as np
import scipy.stats as st

DISP = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen25vl_7b": "Qwen2.5-VL-7B",
        "llava16_13b": "LLaVA-1.6-13B", "llava16_7b": "LLaVA-1.6-7B",
        "internvl3_8b": "InternVL3-8B", "internvl35_8b": "InternVL3.5-8B",
        "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B",
        "gemma3_12b": "Gemma-3-12B"}


def rank(d, key, models):
    order = sorted(models, key=lambda m: d[m][key], reverse=True)
    return {m: i + 1 for i, m in enumerate(order)}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    prompts = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        prompts[name] = json.load(open(path))
    names = list(prompts)
    models = [m for m in DISP if m in prompts[names[0]]]

    print("ABSTENTION RATE by prompt (%)")
    hdr = "  " + f"{'model':14s}" + "".join(f"{n:>8s}" for n in names)
    print(hdr)
    for m in models:
        row = "  " + f"{DISP[m]:14s}"
        for n in names:
            row += f"{100*prompts[n][m]['abstain_rate_clean']:8.1f}"
        print(row)

    print("\nAGGREGATE (VZ-all) RANK by prompt (1 = best)")
    ranks = {n: rank(prompts[n], "vizwiz_clean_all", models) for n in names}
    print(hdr)
    for m in models:
        row = "  " + f"{DISP[m]:14s}"
        for n in names:
            row += f"{ranks[n][m]:8d}"
        print(row)

    print("\nRANKING STABILITY across prompts (Kendall tau on aggregate vs P0)")
    base = names[0]
    bvec = [prompts[base][m]["vizwiz_clean_all"] for m in models]
    for n in names[1:]:
        vec = [prompts[n][m]["vizwiz_clean_all"] for m in models]
        if len(models) < 3:
            print(f"   {base} vs {n}: n/a (need >=3 models)")
            continue
        t, p = st.kendalltau(bvec, vec)
        if t != t:      # nan (e.g. all-tied)
            print(f"   {base} vs {n}: tau=n/a")
        else:
            print(f"   {base} vs {n}: tau={t:.3f}  p={p:.3f}")

    print("\nANSWERING ABILITY (answerable subset) by prompt — should be STABLE if")
    print("the prompt only moves abstention, not perception:")
    print(hdr)
    for m in models:
        row = "  " + f"{DISP[m]:14s}"
        for n in names:
            row += f"{prompts[n][m]['vizwiz_clean_answerable']:8.3f}"
        print(row)

    # headline read
    print("\nREAD: if abstention rate moves a lot across prompts while answerable")
    print("accuracy stays put, the aggregate ranking is driven by the abstention")
    print("instruction (supports Major 3.1). If the aggregate ranking itself is")
    print("stable across prompts, the confound is robust to prompt wording.")


if __name__ == "__main__":
    main()
