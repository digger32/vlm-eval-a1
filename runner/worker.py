#!/usr/bin/env python
"""Run ONE unit. Invoked as a subprocess by orchestrate.py (so a hung unit can be killed
without taking down the batch).

Contract:
  - writes item rows incrementally to results/<uid>.items.jsonl  (resume granularity)
  - on completion writes results/<uid>.json atomically (summary + meta + peak VRAM)
  - exits 0 on success, non-zero on failure (orchestrator logs and moves on)

Usage:
  python -m runner.worker --run-config configs/a1_acvr.yaml \
      --models-config configs/models.yaml --uid a1__qwen3vl_8b__vizwiz_val__blur_s3__seed0
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback

from harness.registry import load_models, load_generation_defaults
from harness.vlm import VLM
from harness import parsing
from harness.corruptions import condition_transform
from data import datasets as dsets
from runner.units import condition_base


# ----------------------------------------------------------- prompt builders
DEFAULT_VQA_PROMPT = ("Answer in a few words. If the image does not allow answering, "
                      "reply 'unanswerable'.")


def build_prompt(item, condition, vqa_prompt=None):
    if item["mcq"]:
        letters = "ABCDEFGH"
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(item["options"]))
        return (f"{item['question']}\n{opts}\n"
                "Answer with the option's letter only.")
    # open-ended VQA (instruction overridable from the run config: `prompt:`)
    return f"{item['question']}\n" + (vqa_prompt or DEFAULT_VQA_PROMPT)


# ----------------------------------------------------- option permutation (MCQ order-robustness)
LETTERS = "ABCDEFGH"


def resolve_gt_index(item):
    """Map item['answer'] -> 0-based ORIGINAL option index, robustly.
    Accepts a letter ('A'..), an int index, or the option text. Raises if unresolvable
    (so the item becomes an error row rather than being silently mis-scored)."""
    a = item["answer"]
    opts = item["options"]
    n = len(opts)
    if isinstance(a, bool):
        raise ValueError(f"answer is bool: {a!r}")
    if isinstance(a, int):
        if 0 <= a < n:
            return a
    if isinstance(a, str):
        s = a.strip()
        if len(s) == 1 and s.upper() in LETTERS[:n]:
            return LETTERS.index(s.upper())
        for i, o in enumerate(opts):
            if o is not None and s.lower() == str(o).strip().lower():
                return i
    raise ValueError(f"cannot resolve gt answer {a!r} among {n} options")


def perm_order(item, condition, seed):
    """Deterministic option order (list of original indices) for a non-circular unit.
    Keyed on (item id, condition, seed) so perm_rand_3 is a fixed, reproducible shuffle
    and identical across re-runs."""
    import random
    n = len(item["options"])
    base = condition_base(condition)
    if base.startswith("perm_rand"):
        rng = random.Random(hash((item["id"], condition, seed)) & 0xFFFFFFFF)
        order = list(range(n))
        rng.shuffle(order)
        return order
    return list(range(n))            # perm0 / identity


def chosen_original_index(pred_letter, order):
    """Map the model's chosen LETTER (permuted space) back to the ORIGINAL option index,
    so we can measure whether the choice follows CONTENT or POSITION across permutations."""
    if not pred_letter or pred_letter not in LETTERS[:len(order)]:
        return None
    return order[LETTERS.index(pred_letter)]


# ----------------------------------------------------------------- main
def run_unit(uid, run_cfg, models_cfg_path, results_dir=None):
    paper, model, dataset, condition, seedtag = uid.split("__")
    seed = int(seedtag.replace("seed", ""))

    results_dir = results_dir or run_cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    items_path = os.path.join(results_dir, f"{uid}.items.jsonl")
    final_path = os.path.join(results_dir, f"{uid}.json")

    # resume: only items with a REAL score count as done; error rows are retried so a
    # transient failure (e.g. an image not yet downloaded) does not get frozen as done.
    done_ids = set()
    if os.path.exists(items_path):
        for line in open(items_path):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "error" not in row and ("acc" in row or "correct" in row):
                done_ids.add(row["id"])

    # dataset + condition transform
    dcfg = run_cfg["datasets"][dataset]
    loader = dsets.get_loader(dcfg["loader"])
    items = loader(split=dcfg.get("split"), max_items=dcfg.get("max_items"), seed=seed)
    cond_spec = run_cfg["conditions"][condition_base(condition)]
    img_tf = condition_transform(cond_spec)
    drop_image = bool(cond_spec.get("drop_image", False))

    # optional per-condition subset (e.g. corruption sweep on a fixed 1500-item subset).
    # deterministic prefix of the already-seed-sampled list -> stays a subset of `clean`,
    # so paired McNemar (clean vs corrupted) remains valid.
    subset_n = cond_spec.get("subset_items")
    if subset_n:
        items = items[:subset_n]

    # model
    specs = load_models(models_cfg_path)
    gdef = load_generation_defaults(models_cfg_path)
    vlm = VLM(specs[model], gdef, item_timeout_s=120)

    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_correct = n_total = 0
    with open(items_path, "a") as fout:
        for item in items:
            if item["id"] in done_ids:
                continue
            try:
                if item["mcq"]:
                    gt_idx = resolve_gt_index(item)
                    image = None if drop_image else dsets.ensure_image(item)
                    if img_tf and image is not None:
                        image = img_tf(image)
                    n = len(item["options"])
                    if condition_base(condition) == "circular":
                        # CircularEval: rotate options through EVERY position; the item
                        # is correct only if the model is right under all rotations.
                        rots = []
                        for shift in range(n):
                            order = [(i + shift) % n for i in range(n)]
                            opts = [item["options"][i] for i in order]
                            gt_letter = LETTERS[order.index(gt_idx)]
                            out = vlm.ask(image, build_prompt({**item, "options": opts},
                                                              condition))
                            pred = parsing.parse_mcq(out, opts)
                            rots.append({"shift": shift, "pred": pred, "gt": gt_letter,
                                         "correct": float(pred == gt_letter),
                                         "chosen_orig": chosen_original_index(pred, order)})
                        correct = float(all(r["correct"] == 1.0 for r in rots))
                        row = {"id": item["id"], "mode": "circular", "gt_orig": gt_idx,
                               "circular_correct": correct, "correct": correct,
                               "n_rot": n, "rotations": rots}
                    else:
                        order = perm_order(item, condition, seed)
                        opts = [item["options"][i] for i in order]
                        gt_letter = LETTERS[order.index(gt_idx)]
                        out = vlm.ask(image, build_prompt({**item, "options": opts},
                                                          condition))
                        pred = parsing.parse_mcq(out, opts)
                        correct = float(pred == gt_letter)
                        row = {"id": item["id"], "pred": pred, "gt": gt_letter,
                               "correct": correct, "order": order, "gt_orig": gt_idx,
                               "chosen_orig": chosen_original_index(pred, order),
                               "raw": out[:200]}
                else:
                    image = None if drop_image else dsets.ensure_image(item)
                    if img_tf and image is not None:
                        image = img_tf(image)
                    out = vlm.ask(image, build_prompt(item, condition, run_cfg.get("prompt")))
                    acc = parsing.vqa_accuracy(out, item["answer"])
                    correct = acc
                    row = {"id": item["id"], "pred": out[:200], "acc": acc,
                           "abstain": parsing.is_unanswerable(out)}
                n_correct += correct
                n_total += 1
            except Exception as e:                       # one bad item != dead unit
                row = {"id": item["id"], "error": repr(e)}
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    summary = {
        "uid": uid, "paper": paper, "model": model, "dataset": dataset,
        "condition": condition, "seed": seed,
        "n_total": n_total, "score": (n_correct / n_total) if n_total else None,
        "wall_s": round(time.time() - t0, 1),
        "peak_vram_gb": round(vlm.peak_vram_bytes() / 1e9, 2),
        "items_file": items_path,
    }
    tmp = final_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, final_path)            # atomic
    print(json.dumps(summary))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--uid", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    import yaml
    run_cfg = yaml.safe_load(open(args.run_config))
    try:
        return run_unit(args.uid, run_cfg, args.models_config, args.results_dir)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
