#!/usr/bin/env python
"""Re-score saved predictions under STRICT (standard, stored) vs LENIENT (containment +
yes/no) VQA accuracy, on both datasets, and report the [strict, lenient] bracket plus
ranking stability (Kendall tau). Offline, no GPU, no re-inference.

Shows that the verbose-model penalty under exact-match is a protocol confound: e.g. Gemma
answers "Yes, it is sweet." (strict 0) when GT is "yes" (lenient 1).

  python -m analysis.rescore --results results --datasets-root datasets \
      [--config configs/a1_acvr.yaml]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
from scipy import stats as sps

from analysis import load, answerability as ans, scoring

VIZWIZ, VQAV2 = "vizwiz_val", "vqav2_val"


def _bracket(rich_cell, gt_map, restrict_answerable=None, ans_meta=None):
    """Return (strict_mean, lenient_mean, tokenf1_mean, n) over a cell, optionally
    restricted to (un)answerable items (VizWiz)."""
    strict, lenient, f1 = [], [], []
    for iid, v in rich_cell.items():
        if restrict_answerable is not None and ans_meta is not None:
            m = ans_meta.get(iid)
            if m is None or m["answerable"] != restrict_answerable:
                continue
        gts = gt_map.get(iid, [])
        strict.append(v["score"])
        lenient.append(scoring.lenient_vqa(v["pred"], gts))
        f1.append(scoring.tokenf1_vqa(v["pred"], gts))
    if not strict:
        return (None, None, None, 0)
    return (float(np.mean(strict)), float(np.mean(lenient)),
            float(np.mean(f1)), len(strict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/out_split/a1_bracket.json")
    args = ap.parse_args()

    rich = load.load_items_rich(args.results, "a1")
    vz_meta = ans.load_vizwiz_meta(args.datasets_root)
    vz_gt = {i: m["gt"] for i, m in vz_meta.items()}
    vq_gt = scoring.load_vqav2_gt(args.datasets_root)
    if not vq_gt:
        print("WARNING: VQAv2 annotations not found under datasets/vqav2 — "
              "VQAv2 lenient will be empty. Finish download_a1 first.")
    models = sorted({k[0] for k in rich})

    def cell(m, ds, c):
        return rich.get((m, ds, c, args.seed), {})

    rows = {}
    print(f"\n=== STRICT vs LENIENT vs TOKEN-F1 bracket ({len(models)} models) ===")
    hdr = (f"{'model':14s} | {'VQAv2 str':>9s} {'len':>6s} {'F1':>6s} | "
           f"{'VZ-ans str':>10s} {'len':>6s} {'F1':>6s}")
    print(hdr); print("-" * len(hdr))
    for m in models:
        vq_s, vq_l, vq_f, _ = _bracket(cell(m, VQAV2, "clean"), vq_gt)
        va_s, va_l, va_f, _ = _bracket(cell(m, VIZWIZ, "clean"), vz_gt,
                                       restrict_answerable=True, ans_meta=vz_meta)
        rows[m] = {"vqav2_strict": vq_s, "vqav2_lenient": vq_l, "vqav2_f1": vq_f,
                   "vizwiz_ans_strict": va_s, "vizwiz_ans_lenient": va_l,
                   "vizwiz_ans_f1": va_f}
        print(f"{m:14s} | {_p(vq_s,9)} {_p(vq_l,6)} {_p(vq_f,6)} | "
              f"{_p(va_s,10)} {_p(va_l,6)} {_p(va_f,6)}")

    # ranking stability: Kendall tau between matcher orderings. The headline check is
    # strict-vs-F1 (two DEFENSIBLE metrics), alongside the original strict-vs-lenient.
    for field in ("vqav2", "vizwiz_ans"):
        ms = [m for m in models if rows[m][f"{field}_strict"] is not None]
        if len(ms) >= 3:
            s = [rows[m][f"{field}_strict"] for m in ms]
            l = [rows[m][f"{field}_lenient"] for m in ms]
            f = [rows[m][f"{field}_f1"] for m in ms]
            tsl, psl = sps.kendalltau(s, l)
            tsf, psf = sps.kendalltau(s, f)
            tlf, plf = sps.kendalltau(l, f)
            print(f"\nranking stability {field} ({len(ms)} models):")
            print(f"   tau(strict, lenient) = {tsl:.3f} (p={psl:.3f})")
            print(f"   tau(strict, tokenF1) = {tsf:.3f} (p={psf:.3f})   <- two defensible metrics")
            print(f"   tau(lenient,tokenF1) = {tlf:.3f} (p={plf:.3f})")
            rows.setdefault("_stability", {})[field] = {
                "tau_strict_lenient": [float(tsl), float(psl)],
                "tau_strict_f1": [float(tsf), float(psf)],
                "tau_lenient_f1": [float(tlf), float(plf)]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")
    print("Bracket = [strict exact-match (lower), lenient containment (upper)]. "
          "Large gap = verbosity-sensitive scoring (e.g. Gemma).")


def _p(x, w):
    return (f"{x:.3f}" if x is not None else "n/a").rjust(w)


if __name__ == "__main__":
    main()
