"""Load item-level results and expose paired / unpaired score vectors.

Each worker wrote results/<uid>.items.jsonl with one row per item:
  open-ended (VQA): {"id","pred","acc","abstain"}
  MCQ           : {"id","pred","gt","correct"}
Per-item score = acc if present else correct. 'error' rows are dropped.

uid = {paper}__{model}__{dataset}__{condition}__seed{seed}
"""
from __future__ import annotations
import glob
import json
import os
from collections import defaultdict

import numpy as np


def _score(row):
    if "error" in row:
        return None
    if "acc" in row:
        return float(row["acc"])
    if "correct" in row:
        return float(row["correct"])
    return None


def load_items(results_dir="results", paper="a1"):
    """-> dict[(model,dataset,condition,seed)] -> dict[item_id] -> score(float)."""
    rich = load_items_rich(results_dir, paper)
    return {k: {i: v["score"] for i, v in d.items()} for k, d in rich.items()}


def load_items_rich(results_dir="results", paper="a1"):
    """-> dict[key] -> dict[item_id] -> {"score","abstain","pred"}. Error rows dropped."""
    data = defaultdict(dict)
    pat = os.path.join(results_dir, f"{paper}__*.items.jsonl")
    for path in glob.glob(pat):
        base = os.path.basename(path)[:-len(".items.jsonl")]
        try:
            _paper, model, dataset, condition, seedtag = base.split("__")
        except ValueError:
            continue
        seed = int(seedtag.replace("seed", ""))
        key = (model, dataset, condition, seed)
        for line in open(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = _score(row)
            if s is not None:
                data[key][row["id"]] = {
                    "score": s, "abstain": bool(row.get("abstain", False)),
                    "pred": row.get("pred", "")}
    return data


def models_in(data):
    return sorted({k[0] for k in data})


def conditions_in(data, dataset=None):
    return sorted({k[2] for k in data if dataset is None or k[1] == dataset})


def cell(data, model, dataset, condition, seed=0):
    """Return dict[item_id]->score for one cell, or {} if absent."""
    return data.get((model, dataset, condition, seed), {})


def paired(data, model, dataset, cond_a, cond_b, seed=0):
    """Aligned score arrays over items present in BOTH conditions (same items).
    Use for clean-vs-corrupted and image-vs-blind (paired tests)."""
    a = cell(data, model, dataset, cond_a, seed)
    b = cell(data, model, dataset, cond_b, seed)
    ids = sorted(set(a) & set(b))
    return (np.array([a[i] for i in ids]),
            np.array([b[i] for i in ids]), ids)


def unpaired(data, model, ds_a, cond_a, ds_b, cond_b, seed=0):
    """Two independent score arrays (different items / datasets). Use for the
    VizWiz-vs-VQAv2 distribution-shift gap (unpaired)."""
    a = cell(data, model, ds_a, cond_a, seed)
    b = cell(data, model, ds_b, cond_b, seed)
    return (np.array(list(a.values())), np.array(list(b.values())))
