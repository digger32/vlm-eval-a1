#!/usr/bin/env python
"""Merge results/<uid>.json summaries into one CSV for stats/plotting.

Usage:
  python -m runner.merge --results results --paper a1 --out results_a1.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os


FIELDS = ["uid", "paper", "model", "dataset", "condition", "seed",
          "n_total", "score", "wall_s", "peak_vram_gb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results, f"{args.paper}__*.json"))):
        if path.endswith(".items.jsonl"):
            continue
        try:
            rows.append(json.load(open(path)))
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    done = sum(1 for r in rows if r.get("score") is not None)
    gpu_h = sum(r.get("wall_s", 0) for r in rows) / 3600
    print(f"merged {len(rows)} units ({done} scored) -> {args.out}; "
          f"total compute {gpu_h:.1f} GPU-h")


if __name__ == "__main__":
    raise SystemExit(main())
