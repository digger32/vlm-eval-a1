"""Figures for A1/A2. Saves PDF (vector, for LaTeX) + PNG (preview).

  degradation_curves : per corruption family, accuracy vs severity, per model (CI band)
  bar_with_ci        : accuracy per model per condition with bootstrap CIs
  cd_diagram         : classic Demsar critical-difference diagram (avg ranks + CD bar)
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def bar_with_ci(rows, path, title=""):
    """rows: list of dicts {label, mean, lo, hi}. One grouped bar chart."""
    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    lo = [r["mean"] - r["lo"] for r in rows]
    hi = [r["hi"] - r["mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.7), 4))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=[lo, hi], capsize=3, color="#4477aa")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]


def degradation_curves(curves, path, title="", xlabel="severity", split_outlier=True):
    """curves: dict[model] -> list of (severity, mean, lo, hi). One line+band per model.

    If a single model sits far below the rest (e.g. a degenerate exact-match outlier),
    the y-axis is broken into a tall bulk panel and a short outlier panel so the
    near-flat majority curves stay legible. Distinct colour+marker per model.
    """
    models = sorted(curves)
    cmap = plt.get_cmap("tab10")
    mean_of = {m: [p[1] for p in sorted(curves[m])] for m in models}

    # detect a low outlier: lowest-mean model clearly separated from the bulk
    do_split = False
    if split_outlier and len(models) > 2:
        lo_m = min(models, key=lambda m: np.mean(mean_of[m]))
        bulk = [v for m in models if m != lo_m for v in mean_of[m]]
        lo_vals = mean_of[lo_m]
        spread = (max(bulk) - min(bulk)) if bulk else 0.0
        gap = (min(bulk) - max(lo_vals)) if bulk else 0.0
        do_split = bool(bulk) and gap > 0.05 and gap > 0.4 * (spread + 1e-9)

    def _plot(ax):
        for i, m in enumerate(models):
            pts = sorted(curves[m])
            xs = [p[0] for p in pts]
            ms = [p[1] for p in pts]
            los = [p[2] for p in pts]
            his = [p[3] for p in pts]
            c = cmap(i % 10)
            ax.plot(xs, ms, marker=_MARKERS[i % len(_MARKERS)], color=c,
                    label=m, lw=1.3, ms=4)
            ax.fill_between(xs, los, his, alpha=0.12, color=c)
        ax.grid(alpha=0.3)

    if do_split:
        fig, (axt, axb) = plt.subplots(
            2, 1, sharex=True, figsize=(6.5, 4.6),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
        _plot(axt); _plot(axb)
        pad_b = 0.4 * (spread + 1e-9)
        axt.set_ylim(min(bulk) - pad_b, max(bulk) + pad_b)
        pad_l = 0.5 * (max(lo_vals) - min(lo_vals) + 1e-9) + 0.02
        axb.set_ylim(min(lo_vals) - pad_l, max(lo_vals) + pad_l)
        # broken-axis cosmetics
        axt.spines["bottom"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axt.tick_params(labelbottom=False)
        d = 0.008
        for ax, ys in ((axt, (-d, +d)), (axb, (1 - d, 1 + d))):
            kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.8)
            ax.plot((-d, +d), ys, **kw)
            ax.plot((1 - d, 1 + d), ys, **kw)
        axb.set_xlabel(xlabel)
        axt.set_ylabel("accuracy")
        axt.yaxis.set_label_coords(-0.085, 0.35)
        axt.set_title(title)
        axt.legend(fontsize=7, ncol=2, loc="best")
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _plot(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def cd_diagram(names, avg_ranks, cd, path, title=""):
    """Classic critical-difference diagram. Lower rank = better (left)."""
    names = list(names)
    avg_ranks = np.asarray(avg_ranks, float)
    order = np.argsort(avg_ranks)
    names = [names[i] for i in order]
    ranks = avg_ranks[order]
    k = len(names)
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.3 * k))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_xaxis()
    ax.hlines(k + 0.5, lo, hi, color="k")
    for r in range(lo, hi + 1):
        ax.vlines(r, k + 0.4, k + 0.6, color="k")
        ax.text(r, k + 0.8, str(r), ha="center", fontsize=8)
    for i, (n, r) in enumerate(zip(names, ranks)):
        y = k - i
        ax.plot([r, r], [y, k + 0.5], color="gray", lw=0.8)
        ax.plot([r, lo - 0.4 if i < k / 2 else hi + 0.4], [y, y], color="gray", lw=0.8)
        side = lo - 0.45 if i < k / 2 else hi + 0.45
        ha = "right" if i < k / 2 else "left"
        ax.text(side, y, f"{n} ({r:.2f})", ha=ha, va="center", fontsize=8)
    # CD bar
    ax.hlines(0.4, hi, hi - cd, color="k", lw=2)
    ax.text(hi - cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path)
