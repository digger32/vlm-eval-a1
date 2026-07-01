#!/usr/bin/env python
"""A1 / П3 #5 — decompose the curated->real gap into a question/language-prior axis
and an image-perception axis (the self-review's Major 3.5).

The headline curated-to-real gap on answerable questions,
    gap = VQAv2_sighted - VizWiz_ans_sighted,
conflates (a) VQAv2 questions/answers being easier from text alone and (b) the image
being more useful on curated photos. A blind text-only control on BOTH datasets
separates them exactly:

    gap = [VQAv2_blind - VizWiz_blind]                                  (question/prior axis)
        + [(VQAv2_sighted-VQAv2_blind) - (VizWiz_sighted-VizWiz_blind)] (image axis)

VizWiz clean+blind and VQAv2 clean are already in results_final; this needs the VQAv2
blind run (configs/a1_imgq.yaml -> results_imgq). Offline, no GPU.

IMPORTANT — abstention confound: the blind cells must be produced under a NO-ABSTENTION
prompt. Under the default prompt ("...reply 'unanswerable'"), an imageless model
abstains, which scores 0 on VQAv2 (no unanswerable credit) and corrupts the blind
floor. Use configs/a1_imgq2.yaml (prompt P1, all four cells in one dir) and pass that
single dir:

  python -m analysis.decompose_axes --results results_imgq2 --datasets-root datasets
"""
from __future__ import annotations
import argparse
import numpy as np

from analysis import load, answerability as ans

VIZWIZ, VQAV2 = "vizwiz_val", "vqav2_val"
DISP = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen25vl_7b": "Qwen2.5-VL-7B",
        "llava16_13b": "LLaVA-1.6-13B", "llava16_7b": "LLaVA-1.6-7B",
        "internvl3_8b": "InternVL3-8B", "internvl35_8b": "InternVL3.5-8B",
        "pixtral_12b": "Pixtral-12B", "idefics3_8b": "Idefics3-8B",
        "gemma3_12b": "Gemma-3-12B"}


def _mean(rich, key, meta=None, answerable=True):
    d = rich.get(key, {})
    if meta is None:
        vals = [v["score"] for v in d.values()]
    else:
        vals = [v["score"] for i, v in d.items()
                if meta.get(i, {}).get("answerable", True) == answerable]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results_final", "results_imgq"],
                    help="one or more results dirs; cells are searched across all "
                         "(later dirs win on collision). Pass a single P1 dir with all "
                         "four cells, or results_final + results_imgq for the split run.")
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rich = {}
    for d in a.results:
        rich.update(load.load_items_rich(d, "a1"))
    meta = ans.load_vizwiz_meta(a.datasets_root)
    models = sorted({k[0] for k in rich})

    rf = rb = rich  # all cells looked up in the merged dict

    rows, agg = {}, {"gap": [], "q": [], "img": []}
    print(f"\n=== Image-axis vs question-axis decomposition ({len(models)} models) ===")
    hdr = (f"{'model':14s} {'VQAv2_s':>8s} {'VZ_s':>7s} {'VQAv2_b':>8s} {'VZ_b':>7s} "
           f"{'gap':>7s} {'qaxis':>7s} {'imgaxis':>8s}")
    print(hdr); print("-" * len(hdr))
    for m in models:
        vq_s = _mean(rf, (m, VQAV2, "clean", a.seed))
        vq_b = _mean(rb, (m, VQAV2, "blind", a.seed))
        vz_s = _mean(rf, (m, VIZWIZ, "clean", a.seed), meta, True)
        vz_b = _mean(rf, (m, VIZWIZ, "blind", a.seed), meta, True)
        if any(np.isnan(x) for x in (vq_s, vq_b, vz_s, vz_b)):
            print(f"{DISP.get(m, m):14s} incomplete (missing a cell) -> skipped")
            continue
        gap = vq_s - vz_s
        q_axis = vq_b - vz_b
        img_axis = (vq_s - vq_b) - (vz_s - vz_b)
        rows[m] = {"vqav2_sighted": vq_s, "vqav2_blind": vq_b,
                   "vizwiz_ans_sighted": vz_s, "vizwiz_ans_blind": vz_b,
                   "gap": gap, "question_axis": q_axis, "image_axis": img_axis}
        agg["gap"].append(gap); agg["q"].append(q_axis); agg["img"].append(img_axis)
        print(f"{DISP.get(m, m):14s} {vq_s:8.3f} {vz_s:7.3f} {vq_b:8.3f} {vz_b:7.3f} "
              f"{gap:+7.3f} {q_axis:+7.3f} {img_axis:+8.3f}")

    if agg["gap"]:
        g, q, im = (np.mean(agg["gap"]), np.mean(agg["q"]), np.mean(agg["img"]))
        # bootstrap over models (resample the model set) -> CI on the aggregate axes:
        # how stable the decomposition is across models, not item noise.
        rng = np.random.default_rng(0)
        G, Q, IM = np.array(agg["gap"]), np.array(agg["q"]), np.array(agg["img"])
        n = len(G)
        bg, bq, bi = [], [], []
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            bg.append(G[idx].mean()); bq.append(Q[idx].mean()); bi.append(IM[idx].mean())
        ci = lambda b: (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        cg, cq, cim = ci(bg), ci(bq), ci(bi)
        print("-" * len(hdr))
        print(f"{'MEAN':14s} {'':8s} {'':7s} {'':8s} {'':7s} "
              f"{g:+7.3f} {q:+7.3f} {im:+8.3f}")
        print(f"\nMean curated-to-real gap = {g:+.3f}  95% CI over models "
              f"[{cg[0]:+.3f}, {cg[1]:+.3f}]")
        print(f"  question/prior axis = {q:+.3f}  [{cq[0]:+.3f}, {cq[1]:+.3f}]")
        print(f"  image axis          = {im:+.3f}  [{cim[0]:+.3f}, {cim[1]:+.3f}]")
        share = 100 * q / g if abs(g) > 1e-9 else float("nan")
        print(f"  (question/prior axis is {share:.0f}% of the gap)")
        print("READ: large question/prior axis -> the gap is mostly easier VQAv2 "
              "questions/answers from text alone; large image axis -> the image is "
              "less useful on BLV photos. Blind cells MUST come from a no-abstention "
              "prompt, else VQAv2-blind is abstention-floored near 0 and the split is "
              "meaningless.")
    return rows


if __name__ == "__main__":
    main()
