"""Unified dataset loaders. Every loader yields item dicts with a common schema:

    {
      "id":       str,                # stable unique id within the dataset/split
      "image":    PIL.Image | None,   # None for text-only / blind condition handled later
      "image_path": str | None,
      "question": str,
      "options":  list[str] | None,   # MCQ options (None for open-ended VQA)
      "answer":   str | list[str],    # gt letter (MCQ) or list of human answers (VQA)
      "mcq":      bool,
    }

Paths assume `download_a1.py` / `download_a2.py` have populated ./datasets/.
Fill any TODO marked with the exact field names once you have peeked at the real files
(every public release names columns slightly differently).
"""
from __future__ import annotations
import glob
import ast
import json
import os
import random
from PIL import Image

DATA_ROOT = os.environ.get("VLM_DATA_ROOT", "datasets")


# ----------------------------------------------------------------- A1 loaders
def _resolve_vizwiz_paths(root, split):
    """VizWiz zips extract differently across mirrors. Find the annotations json and
    the image directory among the likely candidates."""
    ann_cands = [os.path.join(root, "Annotations", f"{split}.json"),
                 os.path.join(root, f"{split}.json"),
                 os.path.join(root, "annotations", f"{split}.json")]
    ann_path = next((p for p in ann_cands if os.path.exists(p)), None)
    if ann_path is None:
        hits = glob.glob(os.path.join(root, "**", f"{split}.json"), recursive=True)
        ann_path = hits[0] if hits else ann_cands[0]
    # probe image dir using the first annotation's filename
    first_img = json.load(open(ann_path))[0]["image"]
    img_cands = [os.path.join(root, split), root,
                 os.path.join(root, split, split), os.path.join(root, "images", split)]
    img_dir = next((d for d in img_cands
                    if os.path.exists(os.path.join(d, first_img))), None)
    if img_dir is None:
        hits = glob.glob(os.path.join(root, "**", first_img), recursive=True)
        img_dir = os.path.dirname(hits[0]) if hits else os.path.join(root, split)
    return ann_path, img_dir


def load_vizwiz(split="val", max_items=None, seed=0):
    """VizWiz-VQA. val split has public answers; test answers are hidden.
    Record fields: image (filename), question, answers[].answer, answer_type, answerable."""
    root = os.path.join(DATA_ROOT, "vizwiz")
    ann_path, img_dir = _resolve_vizwiz_paths(root, split)
    ann = json.load(open(ann_path))
    items = []
    for q in ann:
        items.append({
            "id": f"vizwiz_{split}_{q['image']}",
            "image": None,
            "image_path": os.path.join(img_dir, q["image"]),
            "question": q["question"],
            "options": None,
            "answer": [a["answer"] for a in q.get("answers", [])],
            "mcq": False,
        })
    return _subsample(items, max_items, seed)


def load_vqav2(split="val", max_items=None, seed=0):
    """VQA v2. Images = COCO val2014. Open-ended."""
    root = os.path.join(DATA_ROOT, "vqav2")
    ques = json.load(open(os.path.join(
        root, f"v2_OpenEnded_mscoco_{split}2014_questions.json")))["questions"]
    anns = json.load(open(os.path.join(
        root, f"v2_mscoco_{split}2014_annotations.json")))["annotations"]
    ans_by_q = {a["question_id"]: a for a in anns}
    img_dir = os.path.join(root, f"{split}2014")
    items = []
    for q in ques:
        a = ans_by_q.get(q["question_id"])
        if a is None:
            continue
        fname = f"COCO_{split}2014_{q['image_id']:012d}.jpg"
        items.append({
            "id": f"vqav2_{split}_{q['question_id']}",
            "image": None,
            "image_path": os.path.join(img_dir, fname),
            "question": q["question"],
            "options": None,
            "answer": [x["answer"] for x in a["answers"]],
            "mcq": False,
        })
    return _subsample(items, max_items, seed)


# ----------------------------------------------------------------- A2 loaders
def load_mmmu(split="validation", max_items=None, seed=0):
    """MMMU via HF `datasets`. Mixed MCQ/open; we keep MCQ items for permutation work."""
    from datasets import load_dataset, get_dataset_config_names
    subjects = get_dataset_config_names("MMMU/MMMU")
    items = []
    for subj in subjects:
        ds = load_dataset("MMMU/MMMU", subj, split=split)
        for i, ex in enumerate(ds):
            opts = ex.get("options")
            if isinstance(opts, str):
                # MMMU stores options as a Python-list repr ("['a', 'b']", single
                # quotes) -> json.loads fails; literal_eval handles both reprs.
                opts = ast.literal_eval(opts)
            if not opts:            # skip open-ended for the permutation driver
                continue
            items.append({
                "id": f"mmmu_{subj}_{split}_{i}",
                "image": ex.get("image_1"),
                "image_path": None,
                "question": ex["question"],
                "options": list(opts),
                "answer": ex["answer"],     # letter
                "mcq": True,
            })
    return _subsample(items, max_items, seed)


def load_mmbench(split="dev", max_items=None, seed=0):
    """MMBench dev has public answers (test is server-side)."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MMBench", "en", split=split)
    items = []
    for i, ex in enumerate(ds):
        opts = [ex[k] for k in ("A", "B", "C", "D") if ex.get(k) not in (None, "")]
        items.append({
            "id": f"mmbench_{split}_{ex.get('index', i)}",
            "image": ex["image"],
            "image_path": None,
            "question": ex["question"] + (("\n" + ex["hint"]) if ex.get("hint") else ""),
            "options": opts,
            "answer": ex["answer"],
            "mcq": True,
        })
    return _subsample(items, max_items, seed)


def load_seedbench(split="test_img", max_items=None, seed=0):
    """SEED-Bench image part. MCQ with 4 options."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/SEED-Bench", split="test")
    items = []
    for i, ex in enumerate(ds):
        if ex.get("data_type") not in (None, "image"):
            continue
        opts = [ex.get("choice_a"), ex.get("choice_b"),
                ex.get("choice_c"), ex.get("choice_d")]
        opts = [o for o in opts if o is not None]
        items.append({
            "id": f"seed_{ex.get('question_id', i)}",
            "image": ex["image"][0] if isinstance(ex["image"], list) else ex["image"],
            "image_path": None,
            "question": ex["question"],
            "options": opts,
            "answer": ex["answer"],     # letter
            "mcq": True,
        })
    return _subsample(items, max_items, seed)


# ----------------------------------------------------------------- helpers
_LOADERS = {
    "vizwiz": load_vizwiz, "vqav2": load_vqav2,
    "mmmu": load_mmmu, "mmbench": load_mmbench, "seedbench": load_seedbench,
}


def get_loader(name):
    return _LOADERS[name]


def _subsample(items, max_items, seed):
    if max_items is not None and len(items) > max_items:
        rng = random.Random(seed)
        items = rng.sample(items, max_items)
    return items


def ensure_image(item):
    """Lazily load image from path if not already a PIL.Image."""
    if item["image"] is not None:
        return item["image"]
    if item["image_path"]:
        return Image.open(item["image_path"]).convert("RGB")
    return None
