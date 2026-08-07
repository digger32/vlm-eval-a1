"""Camera-ready offline analyses (no GPU, reads the existing per-prediction outputs).

Answers four reviewer requests directly from results that already exist:

  1. answer-type breakdown (VizWiz's own yes/no | number | other scheme) vs the
     answerability split, to position the protocol against the native benchmark axes;
  2. answerable-subset accuracy under every prompt P0-P3, the number a reviewer flagged
     as claimed but not reported;
  3. rankings by aggregate, by answerable-only, and by correct abstention, with the
     Kendall-tau between them, to show how much of the aggregate order is abstention;
  4. coverage-risk operating points (each prompt is one operating point on the
     abstention/coverage trade-off).

Usage:
  python -m analysis.cr_offline --results-final results_final \
      --psweep results_psweep_P0 results_psweep_P1 results_psweep_P2 results_psweep_P3 \
      --datasets-root datasets
"""
from __future__ import annotations
import argparse, glob, json, os
from itertools import combinations

DISP = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen25vl_7b": "Qwen2.5-VL-7B",
        "llava16_7b": "LLaVA-1.6-7B", "llava16_13b": "LLaVA-1.6-13B",
        "internvl3_8b": "InternVL3-8B", "internvl35_8b": "InternVL3.5-8B",
        "gemma3_12b": "Gemma-3-12B", "pixtral_12b": "Pixtral-12B",
        "idefics3_8b": "Idefics3-8B"}


def find_val_json(root):
    for c in [os.path.join(root, "vizwiz", "Annotations", "val.json"),
              os.path.join(root, "vizwiz", "val.json")]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(root, "vizwiz", "**", "val.json"), recursive=True)
    if not hits:
        raise SystemExit(f"VizWiz val.json not found under {root}")
    return hits[0]


def load_meta(root):
    """id -> {answerable, answer_type}. Item ids are f'vizwiz_val_{image}'."""
    meta = {}
    for r in json.load(open(find_val_json(root))):
        meta[f"vizwiz_val_{r['image']}"] = {
            "answerable": int(r.get("answerable", 0)),
            "answer_type": r.get("answer_type", "other"),
        }
    return meta


def load_units(results_dir, dataset="vizwiz_val", condition="clean"):
    """model -> list of per-item rows for one (dataset, condition) cell."""
    out = {}
    pat = os.path.join(results_dir, f"a1__*__{dataset}__{condition}__seed*.items.jsonl")
    for path in sorted(glob.glob(pat)):
        model = os.path.basename(path).split("__")[1]
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "error" not in r and "acc" in r:
                    rows.append(r)
        if rows:
            out[model] = rows
    return out


def kendall_tau(a, b):
    """Kendall tau-b between two rankings given as {key: score} (higher = better)."""
    keys = sorted(set(a) & set(b))
    conc = disc = ta = tb = 0
    for x, y in combinations(keys, 2):
        da, db = a[x] - a[y], b[x] - b[y]
        if da == 0 and db == 0:
            ta += 1; tb += 1; continue
        if da == 0:
            ta += 1; continue
        if db == 0:
            tb += 1; continue
        if (da > 0) == (db > 0):
            conc += 1
        else:
            disc += 1
    n0 = conc + disc
    denom = ((n0 + ta) * (n0 + tb)) ** 0.5
    return (conc - disc) / denom if denom else float("nan")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def analysis_answer_type(rows_by_model, meta):
    """Accuracy per VizWiz answer_type, plus the answerable/unanswerable split."""
    print("\n" + "=" * 78)
    print("1. VizWiz's own answer-type scheme vs the answerability split")
    print("=" * 78)
    types = ["yes/no", "number", "other"]
    hdr = f"{'model':16s}" + "".join(f"{t:>10s}" for t in types) + \
          f"{'ANSWERABLE':>12s}{'abstain%':>10s}"
    print(hdr); print("-" * len(hdr))
    for m in sorted(rows_by_model, key=lambda k: DISP.get(k, k)):
        rows = rows_by_model[m]
        cells = []
        for t in types:
            v = [r["acc"] for r in rows
                 if meta.get(r["id"], {}).get("answer_type") == t
                 and meta.get(r["id"], {}).get("answerable") == 1]
            cells.append(mean(v))
        ans = mean([r["acc"] for r in rows if meta.get(r["id"], {}).get("answerable") == 1])
        ab = mean([bool(r.get("abstain")) for r in rows])
        print(f"{DISP.get(m,m):16s}" + "".join(f"{c:10.3f}" for c in cells) +
              f"{ans:12.3f}{100*ab:9.1f}%")
    print("\nREAD: the native answer-type axis splits questions by the FORM of the answer;")
    print("it does not separate answering from abstaining, which is what the")
    print("answerability split does. The two axes are complementary, not substitutes.")


def analysis_prompts(psweep_dirs, meta):
    """Answerable-subset accuracy and abstention under each prompt (reviewer request)."""
    print("\n" + "=" * 78)
    print("2. Answerable-subset accuracy under each prompt (was claimed, not reported)")
    print("=" * 78)
    per_prompt = {}
    for d in psweep_dirs:
        tag = os.path.basename(d.rstrip("/")).replace("results_psweep_", "")
        per_prompt[tag] = load_units(d)
    tags = sorted(per_prompt)
    hdr = f"{'model':16s}" + "".join(f"{t+' acc':>10s}{t+' abs%':>10s}" for t in tags)
    print(hdr); print("-" * len(hdr))
    acc_by_tag = {t: {} for t in tags}
    for m in sorted(DISP, key=lambda k: DISP[k]):
        if not any(m in per_prompt[t] for t in tags):
            continue
        line = f"{DISP[m]:16s}"
        for t in tags:
            rows = per_prompt[t].get(m, [])
            a = mean([r["acc"] for r in rows
                      if meta.get(r["id"], {}).get("answerable") == 1])
            ab = mean([bool(r.get("abstain")) for r in rows])
            acc_by_tag[t][m] = a
            line += f"{a:10.3f}{100*ab:9.1f}%"
        print(line)
    print("-" * len(hdr))
    line = f"{'MEAN':16s}"
    for t in tags:
        line += f"{mean(acc_by_tag[t].values()):10.3f}{'':10s}"
    print(line)
    print("\nKendall tau of the ANSWERABLE-ONLY ranking against P0:")
    for t in tags:
        if t == tags[0]:
            continue
        print(f"  tau(P0, {t}) = {kendall_tau(acc_by_tag[tags[0]], acc_by_tag[t]):+.3f}")
    return per_prompt, acc_by_tag


def analysis_ranks(rows_by_model, meta):
    """Aggregate vs answerable-only vs correct-abstention rankings."""
    print("\n" + "=" * 78)
    print("3. What the aggregate ranking is actually ordering")
    print("=" * 78)
    agg, ansacc, corr_abst, abst = {}, {}, {}, {}
    for m, rows in rows_by_model.items():
        agg[m] = mean([r["acc"] for r in rows])
        ansacc[m] = mean([r["acc"] for r in rows
                          if meta.get(r["id"], {}).get("answerable") == 1])
        corr_abst[m] = mean([bool(r.get("abstain")) for r in rows
                             if meta.get(r["id"], {}).get("answerable") == 0])
        abst[m] = mean([bool(r.get("abstain")) for r in rows])
    hdr = (f"{'model':16s}{'VZ-all':>9s}{'VZ-ans':>9s}{'corr-abst':>11s}"
           f"{'abstain%':>10s}{'rk-all':>8s}{'rk-ans':>8s}")
    print(hdr); print("-" * len(hdr))
    rank = lambda d: {m: i + 1 for i, m in
                      enumerate(sorted(d, key=lambda k: -d[k]))}
    r_all, r_ans = rank(agg), rank(ansacc)
    for m in sorted(agg, key=lambda k: -agg[k]):
        print(f"{DISP.get(m,m):16s}{agg[m]:9.3f}{ansacc[m]:9.3f}{corr_abst[m]:11.3f}"
              f"{100*abst[m]:9.1f}%{r_all[m]:8d}{r_ans[m]:8d}")
    print("-" * len(hdr))
    print(f"\n  tau(aggregate, answerable-only)   = {kendall_tau(agg, ansacc):+.3f}")
    print(f"  tau(aggregate, correct-abstention)= {kendall_tau(agg, corr_abst):+.3f}")
    print(f"  tau(aggregate, abstention rate)   = {kendall_tau(agg, abst):+.3f}")
    print(f"  tau(answerable-only, abstention)  = {kendall_tau(ansacc, abst):+.3f}")
    n_moved = sum(1 for m in agg if r_all[m] != r_ans[m])
    print(f"\n  {n_moved}/{len(agg)} models change rank between the aggregate and the")
    print("  answerable-only ordering.")
    return agg, ansacc, corr_abst, abst


def analysis_coverage_risk(per_prompt, meta):
    """Each prompt is an operating point on the coverage/risk trade-off."""
    print("\n" + "=" * 78)
    print("4. Coverage-risk operating points (one per prompt)")
    print("=" * 78)
    print("coverage = fraction of ANSWERABLE questions the model actually attempts;")
    print("risk     = error rate (1 - accuracy) on the attempted answerable questions.\n")
    tags = sorted(per_prompt)
    hdr = f"{'model':16s}" + "".join(f"{t+' cov':>10s}{t+' risk':>10s}" for t in tags)
    print(hdr); print("-" * len(hdr))
    for m in sorted(DISP, key=lambda k: DISP[k]):
        if not any(m in per_prompt[t] for t in tags):
            continue
        line = f"{DISP[m]:16s}"
        for t in tags:
            rows = [r for r in per_prompt[t].get(m, [])
                    if meta.get(r["id"], {}).get("answerable") == 1]
            attempted = [r for r in rows if not r.get("abstain")]
            cov = len(attempted) / len(rows) if rows else float("nan")
            risk = 1.0 - mean([r["acc"] for r in attempted]) if attempted else float("nan")
            line += f"{cov:10.3f}{risk:10.3f}"
        print(line)
    print("\nREAD: a full risk-coverage curve needs a graded confidence score, which the")
    print("released outputs do not carry; the prompt sweep gives a small set of genuine")
    print("operating points instead, which is enough to show the trade-off is real and")
    print("prompt-controllable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-final", default="results_final")
    ap.add_argument("--psweep", nargs="*", default=[])
    ap.add_argument("--datasets-root", default="datasets")
    a = ap.parse_args()

    meta = load_meta(a.datasets_root)
    final = load_units(a.results_final)
    print(f"loaded {len(meta)} annotations, {len(final)} models from {a.results_final}")

    analysis_answer_type(final, meta)
    per_prompt = None
    if a.psweep:
        per_prompt, _ = analysis_prompts(a.psweep, meta)
    analysis_ranks(final, meta)
    if per_prompt:
        analysis_coverage_risk(per_prompt, meta)


if __name__ == "__main__":
    main()
