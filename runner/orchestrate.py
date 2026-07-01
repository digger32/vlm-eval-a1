#!/usr/bin/env python
"""Orchestrate a run: enumerate units, (optionally) skip finished ones, launch each as
its own subprocess with a HARD per-unit wall-clock timeout, log compute, continue on
failure. Emits run_meta.json + manifest.jsonl so the review-proofing gate can verify a
clean final pass (A1).

Usage (inside tmux):
  # incremental / resume (default):
  python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
      --models-config configs/models.yaml
  # FINAL clean pass for frozen numbers (resume OFF, fresh dir):
  python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
      --models-config configs/models.yaml --no-resume --results-dir results_final
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def _env_snapshot():
    snap = {}
    try:
        import torch
        snap["torch"] = torch.__version__
        snap["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            snap["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod in ("transformers", "accelerate"):
        try:
            snap[mod] = __import__(mod).__version__
        except Exception:
            pass
    return snap


def main():
    from runner.units import load_run_config, enumerate_units

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this run")
    ap.add_argument("--dataset", action="append", default=None,
                    help="restrict to these dataset(s); repeatable. Omit = all.")
    ap.add_argument("--no-resume", action="store_true",
                    help="FINAL pass: re-run every unit even if its output exists "
                         "(overwrites). Use with a fresh --results-dir.")
    ap.add_argument("--results-dir", default=None,
                    help="override results_dir from the config (e.g. results_final).")
    args = ap.parse_args()

    cfg = load_run_config(args.run_config)
    results_dir = args.results_dir or cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    timeout_s = int(cfg.get("per_unit_timeout_s", 14400))
    units = enumerate_units(cfg)
    if args.dataset:
        keep = set(args.dataset)
        units = [u for u in units if u.dataset in keep]
        print(f"[orchestrate] dataset filter {keep}: {len(units)} units")

    run_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"paper": cfg["paper"], "run_started": run_started,
            "no_resume": bool(args.no_resume), "results_dir": results_dir,
            "n_units": len(units), "datasets": sorted({u.dataset for u in units}),
            "models": cfg["models"], "env": _env_snapshot(),
            "run_config": os.path.basename(args.run_config)}
    json.dump(meta, open(os.path.join(results_dir, "run_meta.json"), "w"), indent=2)
    manifest = open(os.path.join(results_dir, "manifest.jsonl"), "a")

    log_path = os.path.join(results_dir, f"{cfg['paper']}_orchestration.log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["uid", "status", "wall_s", "rc"])

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    total = len(units)
    print(f"[orchestrate] {total} units; timeout {timeout_s}s; gpu {args.gpu}; "
          f"no_resume={args.no_resume}; dir={results_dir}")

    for i, u in enumerate(units, 1):
        final = os.path.join(results_dir, f"{u.uid}.json")
        if os.path.exists(final) and not args.no_resume:
            print(f"[{i}/{total}] skip (done) {u.uid}")
            manifest.write(json.dumps({"unit": u.uid, "dataset": u.dataset,
                                       "started": run_started, "status": "skip"}) + "\n")
            manifest.flush()
            continue
        print(f"[{i}/{total}] run  {u.uid}")
        cmd = [sys.executable, "-m", "runner.worker",
               "--run-config", args.run_config,
               "--models-config", args.models_config,
               "--uid", u.uid, "--results-dir", results_dir]
        t0 = time.time()
        status, rc = "ok", 0
        try:
            p = subprocess.run(cmd, env=env, timeout=timeout_s)
            rc = p.returncode
            status = "ok" if rc == 0 else "fail"
        except subprocess.TimeoutExpired:
            status, rc = "timeout", -9
        wall = round(time.time() - t0, 1)
        logw.writerow([u.uid, status, wall, rc]); logf.flush()
        manifest.write(json.dumps({"unit": u.uid, "dataset": u.dataset,
                                   "model": u.model, "condition": u.condition,
                                   "seed": u.seed, "started": run_started,
                                   "status": status, "wall_s": wall}) + "\n")
        manifest.flush()
        print(f"     -> {status} ({wall}s)")

    logf.close(); manifest.close()
    print("[orchestrate] complete. Merge with: "
          f"python -m runner.merge --results {results_dir} "
          f"--paper {cfg['paper']} --out results_{cfg['paper']}.csv")


if __name__ == "__main__":
    raise SystemExit(main())
