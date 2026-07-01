"""Lenient (protocol-robust) VQA scoring, as a SECONDARY metric bracketing exact-match.

Motivation: standard VQA exact-match scores a verbose-but-correct answer as 0
("Yes, it is sweet." vs GT "yes"). We report a [strict, lenient] bracket so the verbose
-model penalty is visible and rankings can be checked for stability across protocols.

lenient = VQA-soft accuracy where a GT answer 'matches' if it is a token-substring of the
(normalised) prediction, plus yes/no leading-token extraction. This OVER-credits slightly
(hence an upper bound), so it is never the headline — exact-match stays primary.
"""
from __future__ import annotations
import json
import os
import glob
from collections import Counter

from harness.parsing import _normalize


def _yesno(pred):
    p = _normalize(pred)
    toks = p.split()
    if toks and toks[0] in ("yes", "no"):
        return toks[0]
    return None


def lenient_vqa(pred, gt_answers):
    """VQA-soft accuracy under containment + yes/no extraction. Upper bound on exact."""
    if not pred:
        return 0.0
    p = _normalize(pred)
    pp = " " + p + " "
    yn = _yesno(pred)
    matches = 0
    for g in gt_answers:
        gn = _normalize(g)
        if not gn:
            continue
        if (" " + gn + " ") in pp or (yn is not None and gn == yn):
            matches += 1
    return min(matches / 3.0, 1.0)


def tokenf1_vqa(pred, gt_answers):
    """Normalised token-F1 (SQuAD-style), max over the reference answers.

    A DEFENSIBLE third matcher sitting between strict exact-match and the lenient
    containment upper bound: it rewards partial token overlap, so a verbose-but-correct
    answer ("yes it is sweet" vs GT "yes") scores neither 0 (strict) nor 1 (lenient)
    but a graded value. Used to show the model ranking moves between two reasonable
    metrics, not only between strict and an upper bound. Same normalisation as the
    primary metric (harness.parsing._normalize) for consistency.
    """
    if not pred:
        return 0.0
    p = _normalize(pred).split()
    if not p:
        return 0.0
    best = 0.0
    for g in gt_answers:
        gt = _normalize(g).split()
        if not gt:
            continue
        common = sum((Counter(p) & Counter(gt)).values())
        if common == 0:
            continue
        prec, rec = common / len(p), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ----------------------------------------------------------------- GT loaders
def load_vqav2_gt(datasets_root="datasets", split="val"):
    """-> dict[item_id]->[answers]. item_id matches worker: f'vqav2_{split}_{question_id}'."""
    root = os.path.join(datasets_root, "vqav2")
    cands = glob.glob(os.path.join(root, f"v2_mscoco_{split}2014_annotations.json"))
    if not cands:
        cands = glob.glob(os.path.join(root, "**", f"*{split}2014_annotations.json"),
                          recursive=True)
    if not cands:
        return {}
    anns = json.load(open(cands[0]))["annotations"]
    return {f"vqav2_{split}_{a['question_id']}": [x["answer"] for x in a["answers"]]
            for a in anns}
