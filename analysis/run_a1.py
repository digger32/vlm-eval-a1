#!/usr/bin/env python
"""A1 (ACVR) analysis. Reads results/<a1__...>.items.jsonl as they appear and produces:
  * a headline table (per model): VQAv2 acc, VizWiz acc, shift gap (CI + Mann-Whitney),
    blind acc + image-vs-blind paired test, per-corruption drops with Holm-adjusted p;
  * figures: bar-with-CI (VizWiz clean), degradation curves (blur/exposure/crop),
    CD diagram (models ranked across conditions);
  * analysis/a1_stats.json with everything machine-readable.

Safe to run REPEATEDLY while the A1 run is in progress — it just uses whatever units
have completed so far. Run:
  python -m analysis.run_a1 --results results --config configs/a1_acvr.yaml
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import yaml

from analysis import load, stats, figures

VIZWIZ = "vizwiz_val"
VQAV2 = "vqav2_val"


def _fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/a1_acvr.yaml")
    ap.add_argument("--outdir", default="analysis/out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    conds = cfg["conditions"]
    corruption_conds = [c for c in conds
                        if "corruption" in conds[c]]
    data = load.load_items(args.results, paper="a1")
    if not data:
        print("no completed units yet — nothing to analyse.")
        return
    models = load.models_in(data)
    os.makedirs(args.outdir, exist_ok=True)

    report = {"models": {}, "config": os.path.basename(args.config)}
    table_rows = []

    for m in models:
        row = {"model": m}
        # --- accuracies + bootstrap CIs
        vq = np.array(list(load.cell(data, m, VQAV2, "clean", args.seed).values()))
        vz = np.array(list(load.cell(data, m, VIZWIZ, "clean", args.seed).values()))
        row["vqav2_acc"] = stats.bootstrap_ci(vq) if len(vq) else (None,)*3
        row["vizwiz_acc"] = stats.bootstrap_ci(vz) if len(vz) else (None,)*3

        # --- distribution-shift gap (UNPAIRED: different items)
        if len(vq) and len(vz):
            gap = stats.unpaired_bootstrap_diff(vq, vz)        # vqav2 - vizwiz
            _, mw_p = stats.mannwhitney(vq, vz)
            row["shift_gap"] = gap
            row["shift_mw_p"] = mw_p
        else:
            row["shift_gap"], row["shift_mw_p"] = (None,)*3, None

        # --- blind-LLM control (PAIRED: image vs no-image, same VizWiz items)
        a, b, ids = load.paired(data, m, VIZWIZ, "clean", "blind", args.seed)
        if len(ids):
            row["blind_acc"] = float(b.mean())
            row["img_minus_blind"] = stats.paired_bootstrap_diff(a, b)
            row["blind_wilcoxon_p"] = stats.wilcoxon_paired(a, b)[1]
            row["blind_mcnemar"] = stats.mcnemar(a, b)          # (b01,c10,p)
        else:
            row["blind_acc"] = None

        # --- corruption sweep (PAIRED: clean vs corrupted on the subset items)
        corr = {}
        praw = []
        for c in corruption_conds:
            a, b, ids = load.paired(data, m, VIZWIZ, "clean", c, args.seed)
            if not len(ids):
                continue
            drop = stats.paired_bootstrap_diff(a, b)            # clean - corrupted
            _, wp = stats.wilcoxon_paired(a, b)
            corr[c] = {"drop": drop, "wilcoxon_p": wp,
                       "mcnemar": stats.mcnemar(a, b), "n": len(ids)}
            praw.append((c, wp))
        # Holm across the corruption family
        if praw:
            adj = stats.holm([p for _, p in praw])
            for (c, _), pa in zip(praw, adj):
                corr[c]["wilcoxon_p_holm"] = float(pa)
        row["corruptions"] = corr
        report["models"][m] = row
        table_rows.append(row)

    # ---------------- console table
    print(f"\n=== A1 results ({len(models)} models, seed {args.seed}) ===")
    hdr = f"{'model':16s} {'VQAv2':>7s} {'VizWiz':>7s} {'gap':>7s} {'mw_p':>7s} {'blind':>7s} {'i-b_p':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in table_rows:
        gap = r["shift_gap"][0] if r["shift_gap"][0] is not None else None
        bp = r.get("blind_wilcoxon_p")
        print(f"{r['model']:16s} {_fmt(r['vqav2_acc'][0]):>7s} {_fmt(r['vizwiz_acc'][0]):>7s} "
              f"{_fmt(gap):>7s} {_fmt(r['shift_mw_p']):>7s} {_fmt(r['blind_acc']):>7s} {_fmt(bp):>7s}")

    # ---------------- figures
    figdir = os.path.join(args.outdir, "figs")
    # bar: VizWiz clean acc per model
    bar = [{"label": r["model"], "mean": r["vizwiz_acc"][0],
            "lo": r["vizwiz_acc"][1], "hi": r["vizwiz_acc"][2]}
           for r in table_rows if r["vizwiz_acc"][0] is not None]
    if bar:
        figures.bar_with_ci(bar, os.path.join(figdir, "vizwiz_clean_acc"),
                            title="VizWiz (clean) accuracy by model")

    # degradation curves per family
    families = {"blur": [], "exposure": [], "crop": []}
    for c in corruption_conds:
        kind = conds[c]["corruption"]
        fam = "blur" if kind == "gaussian_blur" else ("exposure" if kind == "exposure" else "crop")
        families[fam].append((c, conds[c]["severity"]))
    for fam, items in families.items():
        if not items:
            continue
        curves = {}
        for r in table_rows:
            m = r["model"]
            pts = []
            vz = stats.bootstrap_ci(
                list(load.cell(data, m, VIZWIZ, "clean", args.seed).values()))
            if vz[0] is not None:
                pts.append((0, vz[0], vz[1], vz[2]))
            for c, sev in items:
                vals = list(load.cell(data, m, VIZWIZ, c, args.seed).values())
                if vals:
                    ci = stats.bootstrap_ci(vals)
                    pts.append((sev, ci[0], ci[1], ci[2]))
            if len(pts) > 1:
                curves[m] = pts
        if curves:
            figures.degradation_curves(
                curves, os.path.join(figdir, f"degradation_{fam}"),
                title=f"VizWiz robustness to {fam}", xlabel=f"{fam} severity")

    # CD diagram: models ranked across conditions (blocks)
    cond_list = [c for c in conds if any(
        load.cell(data, m, VIZWIZ, c, args.seed) for m in models)]
    mat = []
    for c in cond_list:
        rowv = [np.mean(list(load.cell(data, m, VIZWIZ, c, args.seed).values()) or [np.nan])
                for m in models]
        if not any(np.isnan(rowv)):
            mat.append(rowv)
    if len(mat) >= 2 and len(models) >= 3:
        fn = stats.friedman_nemenyi(mat)
        report["friedman_nemenyi"] = {
            "friedman_p": fn["friedman_p"], "cd": fn["cd"],
            "avg_ranks": dict(zip(models, fn["avg_ranks"].tolist()))}
        figures.cd_diagram(models, fn["avg_ranks"], fn["cd"],
                           os.path.join(figdir, "cd_models"),
                           title=f"Models ranked across {len(mat)} conditions "
                                 f"(Friedman p={fn['friedman_p']:.1e})")
        print(f"\nFriedman p={fn['friedman_p']:.2e}  CD={fn['cd']:.2f}")

    # ---------------- dump
    with open(os.path.join(args.outdir, "a1_stats.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else o)
    print(f"\nwrote {args.outdir}/a1_stats.json and figures in {figdir}/")


if __name__ == "__main__":
    main()
