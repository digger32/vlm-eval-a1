"""VizWiz answerability + GT helpers for the answerable/unanswerable split.

The split is the core fix: VizWiz val is 32% 'unanswerable', so aggregate accuracy
rewards abstention and masks degradation. We therefore report, per condition:
  * answerable subset   -> real VQA skill (degradation should HURT here)
  * unanswerable subset -> abstention-correct rate (degradation 'helps' here)
  * abstention rate      -> fraction predicting 'unanswerable'
"""
from __future__ import annotations
import json
import os
import glob


def _find_val_json(root):
    cands = [os.path.join(root, "vizwiz", "Annotations", "val.json"),
             os.path.join(root, "vizwiz", "val.json")]
    for p in cands:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(root, "vizwiz", "**", "val.json"), recursive=True)
    return hits[0] if hits else cands[0]


def load_vizwiz_meta(datasets_root="datasets", split="val"):
    """-> dict[item_id] -> {"answerable": bool, "gt": [answers]}.
    item_id matches the worker's id: f'vizwiz_{split}_{image_filename}'."""
    path = _find_val_json(datasets_root)
    meta = {}
    for r in json.load(open(path)):
        iid = f"vizwiz_{split}_{r['image']}"
        meta[iid] = {"answerable": bool(r.get("answerable", 1)),
                     "gt": [a["answer"] for a in r.get("answers", [])]}
    return meta


def relaxed_correct(pred, gt_answers):
    """Containment match: 1.0 if any GT answer appears as a token-substring of pred
    (normalised), else 0.0. SECONDARY metric only — over-credits, so never the headline.
    Used to check whether a verbose model's exact-match losses are format vs real error."""
    if not pred:
        return 0.0
    p = " " + pred.lower().strip().strip(".") + " "
    for g in gt_answers:
        g = g.lower().strip()
        if g and (" " + g + " ") in p:
            return 1.0
    return 0.0
