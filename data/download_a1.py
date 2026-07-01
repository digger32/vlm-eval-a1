#!/usr/bin/env python
"""Download A1 datasets into ./datasets/ — flaky-network safe.

  VizWiz-VQA  val   : images ~3.2GB + annotations (research/non-commercial).
  VQA v2      val   : questions + annotations (visualqa.org S3).
  COCO val2014      : ~6.2GB images for VQA v2.

Robustness:
  * each download retries with resume (`wget -c`) on transient SSL/network failures;
  * a `<zip>.done` marker is written only after a successful unzip, so a partial file is
    never mistaken for complete;
  * VizWiz and VQA are independent — one failing does not abort the other;
  * everything is logged to results/download_<ts>.log (use `tail -f` to watch).

Run:  python -m data.download_a1            # finishes whatever is missing (resume-safe)
      python -m data.download_a1 --only vqav2
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime

ROOT = os.environ.get("VLM_DATA_ROOT", "datasets")

VIZWIZ_URLS = {
    "val.zip": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "Annotations.zip": "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip",
}
VQA_URLS = {
    "v2_Questions_Val_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}
COCO_VAL = {"val2014.zip": "http://images.cocodataset.org/zips/val2014.zip"}

MAX_TRIES = 20
LOG = None


def log(msg):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n"); LOG.flush()


def _wget(url, dst):
    """wget with resume, retrying on non-zero exit (transient SSL/eof)."""
    for attempt in range(1, MAX_TRIES + 1):
        log(f"  [get {attempt}/{MAX_TRIES}] {url}")
        rc = subprocess.call(
            ["wget", "-c", "--tries=3", "--timeout=60",
             "--progress=dot:giga", "-O", dst, url],
            stdout=LOG or sys.stdout, stderr=subprocess.STDOUT)
        if rc == 0:
            return True
        log(f"  [warn] wget rc={rc}; retrying after resume in 10s")
        time.sleep(10)
    log(f"  [FAIL] gave up on {url}")
    return False


def _get(name, url, ddir):
    """Download <name> into <ddir> and unzip, skipping if already .done."""
    os.makedirs(ddir, exist_ok=True)
    z = os.path.join(ddir, name)
    done = z + ".done"
    if os.path.exists(done):
        log(f"  [skip] {name} already done")
        return True
    if not _wget(url, z):
        return False
    try:
        log(f"  [unzip] {name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ddir)
    except zipfile.BadZipFile:
        log(f"  [bad zip] {name} -> deleting partial, will re-download next run")
        os.remove(z)
        return False
    open(done, "w").close()
    return True


def get_vizwiz():
    log("VizWiz-VQA (val) — ~3.2GB, research/non-commercial")
    d = os.path.join(ROOT, "vizwiz")
    return all(_get(n, u, d) for n, u in VIZWIZ_URLS.items())


def get_vqav2():
    log("VQA v2 (val) questions+annotations + COCO val2014 (~6.2GB)")
    d = os.path.join(ROOT, "vqav2")
    ok = all(_get(n, u, d) for n, u in VQA_URLS.items())
    ok = all(_get(n, u, d) for n, u in COCO_VAL.items()) and ok
    return ok


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["vizwiz", "vqav2"], default=None)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG = open(os.path.join("results", f"download_{ts}.log"), "w")
    log(f"download -> results/download_{ts}.log; data root={ROOT}")

    results = {}
    if args.only in (None, "vizwiz"):
        try:
            results["vizwiz"] = get_vizwiz()
        except Exception as e:
            log(f"  [error] vizwiz: {e!r}"); results["vizwiz"] = False
    if args.only in (None, "vqav2"):
        try:
            results["vqav2"] = get_vqav2()
        except Exception as e:
            log(f"  [error] vqav2: {e!r}"); results["vqav2"] = False

    log(f"SUMMARY: {results}")
    log("peek datasets/vizwiz to confirm val.json + image dir; "
        "data/datasets.py auto-resolves the layout.")
    LOG.close()
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
