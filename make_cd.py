#!/usr/bin/env python3
"""
Rebuild the A1 critical-difference diagram from the released clean run.

Reproduces EXACTLY the numbers now in the manuscript (Fig. 2 / Sec. 4.6):
    Friedman chi2 = 59.387,  p = 6.148e-10,  Nemenyi CD = 3.799
over 9 models x 10 (dataset x condition) blocks, ranking by per-condition
aggregate accuracy. If your numbers differ, the input CSV differs from the
released results_a1.csv -- stop and reconcile before freezing.

Usage (instant; no GPU, no tmux needed):
    python3 make_cd.py                          # looks for results_a1.csv nearby
    python3 make_cd.py --csv results_final/results_a1.csv --out figs/cd_models.pdf
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISP = {
    "qwen3vl_8b": "Qwen3-VL-8B", "qwen25vl_7b": "Qwen2.5-VL-7B",
    "llava16_13b": "LLaVA-1.6-13B", "llava16_7b": "LLaVA-1.6-7B",
    "internvl3_8b": "InternVL3-8B", "internvl35_8b": "InternVL3.5-8B",
    "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B",
    "gemma3_12b": "Gemma-3-12B",
}
Q_ALPHA_K9 = 3.102  # Studentised range / sqrt(2), Nemenyi, alpha=0.05, k=9 (Demsar 2006)


def find_csv(explicit):
    if explicit:
        return Path(explicit)
    for c in ["results_a1.csv", "results_final/results_a1.csv",
              "analysis/out_split/results_a1.csv", "out/results_a1.csv"]:
        if Path(c).is_file():
            return Path(c)
    sys.exit("results_a1.csv not found; pass it with --csv PATH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="figs/cd_models.pdf")
    args = ap.parse_args()

    path = find_csv(args.csv)
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(dict)
    for r in rows:
        by[r["model"]][(r["dataset"], r["condition"])] = float(r["score"])

    models = sorted(by)
    blocks = sorted(set(k for m in by for k in by[m]))
    k, N = len(models), len(blocks)
    if k != 9:
        print(f"WARNING: expected 9 models, found {k}: {models}", file=sys.stderr)

    # Friedman over the (dataset x condition) blocks
    data = [[by[m][b] for b in blocks] for m in models]
    chi, p = st.friedmanchisquare(*data)

    # average ranks (1 = best = highest accuracy)
    R = np.zeros(k)
    for b in blocks:
        sc = np.array([by[m][b] for m in models])
        R += st.rankdata(-sc)
    avg = R / N
    CD = Q_ALPHA_K9 * np.sqrt(k * (k + 1) / (6.0 * N))

    print(f"input        : {path}")
    print(f"models x blk : {k} x {N}")
    print(f"Friedman     : chi2={chi:.3f}  p={p:.3e}")
    print(f"Nemenyi CD   : {CD:.3f}")
    pairs = sorted(zip(avg, [DISP.get(m, m) for m in models]))
    for r, n in pairs:
        print(f"   {n:14s} {r:.3f}")

    # ---- Demsar CD diagram ----
    ranks = [r for r, _ in pairs]
    names = [n for _, n in pairs]
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < CD:
            j += 1
        runs.append((i, j))
    cliques = sorted({(a, b) for a, b in runs
                      if not any((A <= a and b <= B) and (A, B) != (a, b) for A, B in runs)})

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_xlim(lo - 0.6, hi + 0.6); ax.set_ylim(-0.05, 1.18); ax.axis("off")
    ya = 0.82
    ax.plot([lo, hi], [ya, ya], "k", lw=1.3)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [ya, ya + 0.035], "k", lw=1.1)
        ax.text(t, ya + 0.075, str(t), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, ya + 0.20, "average rank (1 = best)", ha="center", fontsize=9)
    half = int(np.ceil(k / 2))
    for i in range(k):
        r, n = ranks[i], names[i]
        if i < half:
            yl = ya - 0.12 - 0.115 * i
            ax.plot([r, r], [ya, yl], "k", lw=1.0)
            ax.plot([r, lo - 0.55], [yl, yl], "k", lw=1.0)
            ax.text(lo - 0.6, yl, n, ha="right", va="center", fontsize=9)
        else:
            ii = k - 1 - i; yl = ya - 0.12 - 0.115 * ii
            ax.plot([r, r], [ya, yl], "k", lw=1.0)
            ax.plot([r, hi + 0.55], [yl, yl], "k", lw=1.0)
            ax.text(hi + 0.6, yl, n, ha="left", va="center", fontsize=9)
    ax.plot([lo, lo + CD], [ya + 0.40, ya + 0.40], "k", lw=2.4)
    for xx in (lo, lo + CD):
        ax.plot([xx, xx], [ya + 0.375, ya + 0.425], "k", lw=2.4)
    ax.text(lo + CD / 2, ya + 0.45, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=9)
    for idx, (a, b) in enumerate(cliques):
        yy = ya - 0.05 - 0.05 * idx
        ax.plot([ranks[a] - 0.05, ranks[b] + 0.05], [yy, yy], "k", lw=3.4, solid_capstyle="round")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
