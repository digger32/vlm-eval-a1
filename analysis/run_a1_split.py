#!/usr/bin/env python
"""A1 analysis WITH the answerable/unanswerable split (the honest version).

For VizWiz we split every condition by the dataset 'answerable' flag and report:
  * ANSWERABLE subset   -> real VQA skill; degradation should HURT (the true claim)
  * UNANSWERABLE subset -> abstention-correct rate
  * abstention rate     -> fraction predicting 'unanswerable'
Headline distribution shift = VQAv2(clean) vs VizWiz(clean, ANSWERABLE only).

Runs offline on saved per-item results (no GPU, no re-inference).
  python -m analysis.run_a1_split --results results --config configs/a1_acvr.yaml \
      --datasets-root datasets [--relaxed]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import yaml

from analysis import load, stats, figures, answerability as ans

VIZWIZ, VQAV2 = "vizwiz_val", "vqav2_val"


def _split(cell, meta):
    """cell: dict[id]->{score,abstain,pred}. -> (ans_scores, unans_scores, abstain_rate)."""
    a, u, ab = [], [], []
    for iid, v in cell.items():
        ab.append(v["abstain"])
        m = meta.get(iid)
        if m is None:
            continue
        (a if m["answerable"] else u).append(v["score"])
    return (np.array(a), np.array(u),
            float(np.mean(ab)) if ab else float("nan"))


def _ans_vec(cell, meta, answerable=True):
    """Aligned-by-id score dict restricted to (un)answerable items."""
    return {iid: v["score"] for iid, v in cell.items()
            if meta.get(iid, {}).get("answerable", True) == answerable}


def _paired_on(da, db):
    ids = sorted(set(da) & set(db))
    return np.array([da[i] for i in ids]), np.array([db[i] for i in ids]), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/a1_acvr.yaml")
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--outdir", default="analysis/out_split")
    ap.add_argument("--relaxed", action="store_true",
                    help="also compute containment (relaxed) acc on answerable items")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    conds = cfg["conditions"]
    corr = [c for c in conds if "corruption" in conds[c]]
    rich = load.load_items_rich(args.results, "a1")
    if not rich:
        print("no completed units yet."); return
    meta = ans.load_vizwiz_meta(args.datasets_root)
    models = sorted({k[0] for k in rich})
    os.makedirs(args.outdir, exist_ok=True)

    def cell(m, ds, c):
        return rich.get((m, ds, c, args.seed), {})

    report = {}
    print(f"\n=== A1 (answerable/unanswerable split), {len(models)} models ===")
    hdr = (f"{'model':14s} {'VQAv2':>7s} {'VZ_all':>7s} {'VZ_ans':>7s} {'VZ_unans':>8s} "
           f"{'abst%':>6s} {'shift(ans)':>10s}")
    print(hdr); print("-" * len(hdr))

    for m in models:
        vq = np.array([v["score"] for v in cell(m, VQAV2, "clean").values()])
        vz_all = np.array([v["score"] for v in cell(m, VIZWIZ, "clean").values()])
        a, u, ab = _split(cell(m, VIZWIZ, "clean"), meta)
        shift = (float(vq.mean()) - float(a.mean())) if len(vq) and len(a) else None
        shift_ci = (stats.unpaired_bootstrap_diff(vq, a)
                    if len(vq) and len(a) else None)
        report[m] = {
            "vqav2_clean": float(vq.mean()) if len(vq) else None,
            "vizwiz_clean_all": float(vz_all.mean()) if len(vz_all) else None,
            "vizwiz_clean_answerable": float(a.mean()) if len(a) else None,
            "vizwiz_clean_unanswerable": float(u.mean()) if len(u) else None,
            "abstain_rate_clean": ab,
            "shift_gap_answerable": shift,
            "shift_gap_answerable_ci": shift_ci,  # (mean, lo, hi), 95% bootstrap
        }
        print(f"{m:14s} {_f(vq)} {_f(vz_all)} {_f(a)} {_f(u,8)} {ab*100:6.1f} "
              f"{(f'{shift:+.3f}' if shift is not None else 'n/a'):>10s}")

        # blind effect on ANSWERABLE items (real language-prior leakage)
        ca = _ans_vec(cell(m, VIZWIZ, "clean"), meta, True)
        ba = _ans_vec(cell(m, VIZWIZ, "blind"), meta, True)
        x, y, ids = _paired_on(ca, ba)
        if len(ids):
            report[m]["blind_answerable"] = {
                "clean_acc": float(x.mean()), "blind_acc": float(y.mean()),
                "drop": stats.paired_bootstrap_diff(x, y),
                "wilcoxon_p": stats.wilcoxon_paired(x, y)[1],
                "mcnemar": stats.mcnemar(x, y)}

        # corruption degradation on ANSWERABLE items (should now HURT)
        cc = {}
        praw = []
        for c in corr:
            ca = _ans_vec(cell(m, VIZWIZ, "clean"), meta, True)
            co = _ans_vec(cell(m, VIZWIZ, c), meta, True)
            x, y, ids = _paired_on(ca, co)
            if not len(ids):
                continue
            cc[c] = {"drop": stats.paired_bootstrap_diff(x, y),
                     "wilcoxon_p": stats.wilcoxon_paired(x, y)[1], "n": len(ids)}
            praw.append((c, cc[c]["wilcoxon_p"]))
        if praw:
            adj = stats.holm([p for _, p in praw])
            for (c, _), pa in zip(praw, adj):
                cc[c]["wilcoxon_p_holm"] = float(pa)
        report[m]["corruptions_answerable"] = cc

        if args.relaxed:
            preds = cell(m, VIZWIZ, "clean")
            rel = [ans.relaxed_correct(v["pred"], meta.get(i, {}).get("gt", []))
                   for i, v in preds.items() if meta.get(i, {}).get("answerable")]
            report[m]["vizwiz_clean_answerable_relaxed"] = float(np.mean(rel)) if rel else None

    # degradation curves on ANSWERABLE subset
    figdir = os.path.join(args.outdir, "figs")
    fams = {"blur": "gaussian_blur", "exposure": "exposure", "crop": "center_crop"}
    for fam, kind in fams.items():
        cs = [(c, conds[c]["severity"]) for c in corr if conds[c]["corruption"] == kind]
        if not cs:
            continue
        curves = {}
        for m in models:
            a0, _, _ = _split(cell(m, VIZWIZ, "clean"), meta)
            pts = [(0, float(a0.mean()), float(a0.mean()), float(a0.mean()))] if len(a0) else []
            for c, sev in cs:
                ac, _, _ = _split(cell(m, VIZWIZ, c), meta)
                if len(ac):
                    ci = stats.bootstrap_ci(ac)
                    pts.append((sev, ci[0], ci[1], ci[2]))
            if len(pts) > 1:
                curves[m] = pts
        if curves:
            figures.degradation_curves(
                curves, os.path.join(figdir, f"answerable_{fam}"),
                title=f"VizWiz ANSWERABLE robustness to {fam}", xlabel=f"{fam} severity")

    with open(os.path.join(args.outdir, "a1_split_stats.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {args.outdir}/a1_split_stats.json + figures in {figdir}/")
    print("NOTE: 'VZ_unans' is the correct-abstention rate; 'shift(ans)' is the honest "
          "distribution-shift drop on answerable questions only.")


def _f(x, w=7):
    return (f"{x.mean():.3f}" if len(x) else "n/a").rjust(w)


if __name__ == "__main__":
    main()
