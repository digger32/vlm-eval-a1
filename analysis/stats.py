"""Statistics for the benchmark/critique papers.

Design choices (matter for VQA soft accuracy in {0, .33, .67, 1}):
  * PAIRED comparisons (clean-vs-corrupted, image-vs-blind; SAME items):
      - Wilcoxon signed-rank on per-item score deltas  -> PRIMARY (handles soft scores)
      - McNemar on a 0.5-binarised version             -> SECONDARY (binary view)
  * UNPAIRED comparison (VizWiz-vs-VQAv2; DIFFERENT items):
      - two-sample bootstrap CI on the mean gap + Mann-Whitney U
  * Multiple comparisons: Holm across the corruption family.
  * Ranking >2 models across blocks: Friedman + Nemenyi critical difference.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

_RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(x, n_boot=10000, alpha=0.05, stat=np.mean, rng=_RNG):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    boots = np.array([stat(rng.choice(x, len(x), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(x)), float(lo), float(hi))


def paired_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, rng=_RNG):
    """Mean(a-b) with CI, resampling item indices jointly (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))


def unpaired_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, rng=_RNG):
    """Mean(a)-Mean(b) with CI, resampling each group independently (unpaired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    boots = np.array([a[rng.integers(0, na, na)].mean() - b[rng.integers(0, nb, nb)].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(a.mean() - b.mean()), float(lo), float(hi))


# ---------------------------------------------------------------- paired tests
def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return (np.nan, 1.0)
    try:
        s, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                              alternative="two-sided")
        return (float(s), float(p))
    except ValueError:
        return (np.nan, 1.0)


def mcnemar(a, b, thresh=0.5):
    """McNemar on 0/1-binarised paired scores. Returns (b01, c10, p).
    b01 = a wrong & b right; c10 = a right & b wrong. Exact binomial if b+c<25."""
    a = (np.asarray(a, float) >= thresh).astype(int)
    b = (np.asarray(b, float) >= thresh).astype(int)
    b01 = int(np.sum((a == 0) & (b == 1)))
    c10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + c10
    if n == 0:
        return (b01, c10, 1.0)
    if n < 25:
        p = float(stats.binomtest(min(b01, c10), n, 0.5).pvalue)
    else:
        chi2 = (abs(b01 - c10) - 1) ** 2 / n
        p = float(stats.chi2.sf(chi2, 1))
    return (b01, c10, p)


# ---------------------------------------------------------------- unpaired test
def mannwhitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, 1.0)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (float(u), float(p))


# ---------------------------------------------------------------- corrections
def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- ranking
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164}   # Nemenyi q_0.05 (infinite df)


def friedman_nemenyi(score_matrix, alpha=0.05):
    """score_matrix: rows=blocks (datasets/conditions), cols=models. Higher=better.
    Returns dict with Friedman p, average ranks (lower=better), and Nemenyi CD."""
    M = np.asarray(score_matrix, float)
    N, k = M.shape
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.array([stats.rankdata(-M[i]) for i in range(N)])  # 1=best per block
    avg_ranks = ranks.mean(axis=0)
    q = _Q05.get(k, 3.164)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return {"friedman_chi2": float(chi2), "friedman_p": float(p),
            "avg_ranks": avg_ranks, "cd": float(cd), "N": N, "k": k}
