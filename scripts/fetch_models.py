#!/usr/bin/env python
"""Robust model prefetch for flaky networks.

Problem: `hf download` (and snapshot_download) can stall at 0 B/s on an unstable link and
hang forever. This wrapper watches the repo's cache directory; if its byte size does not
grow for STALL_SECS, it KILLS the download and relaunches it — `hf download` resumes from
the partial files, so no progress is lost. Repeats until the repo finishes.

Everything is logged to results/fetch_<timestamp>.log so you can upload it.

Run (in tmux):
  python -m scripts.fetch_models --models configs/models.yaml
  python -m scripts.fetch_models --models configs/models.yaml --only gemma3_12b molmo7b_d

Notes:
  * hf_transfer is DISABLED here on purpose (it gives less control on flaky links and is a
    common cause of the 0 B/s hang). The watchdog makes plain downloads reliable instead.
  * Gated-but-unapproved repos (e.g. Llama while PENDING) will fail fast and be reported as
    'gave up (auth/gated?)' — that is expected; the rest still download.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

STALL_SECS = 180        # no byte growth for this long -> kill & resume
POLL_SECS = 20          # how often to check directory size
MAX_ATTEMPTS = 40       # per repo before giving up


def hub_cache() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_dir(repo: str) -> Path:
    return hub_cache() / ("models--" + repo.replace("/", "--"))


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def log(msg: str, fh):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def fetch_one(repo: str, fh) -> bool:
    cache = repo_cache_dir(repo)
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"[{repo}] attempt {attempt} (cache so far {dir_size(cache)/1e9:.2f} GB)", fh)
        proc = subprocess.Popen(
            ["hf", "download", repo],
            stdout=fh, stderr=subprocess.STDOUT, env=env)
        last_size, last_change = -1, time.time()
        killed_for_stall = False
        while proc.poll() is None:
            time.sleep(POLL_SECS)
            sz = dir_size(cache)
            if sz > last_size:
                last_size, last_change = sz, time.time()
            elif time.time() - last_change > STALL_SECS:
                log(f"[{repo}] STALL ({STALL_SECS}s no growth) -> kill & resume", fh)
                proc.terminate()
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                killed_for_stall = True
                break
        rc = proc.wait()
        if rc == 0 and not killed_for_stall:
            log(f"[{repo}] DONE ({dir_size(cache)/1e9:.2f} GB)", fh)
            return True
        if not killed_for_stall and rc != 0:
            # fast non-zero exit with little/no data -> likely gated/unauthorized
            if dir_size(cache) < 1e6 and attempt >= 3:
                log(f"[{repo}] gave up (auth/gated? rc={rc})", fh)
                return False
        time.sleep(5)
    log(f"[{repo}] gave up after {MAX_ATTEMPTS} attempts", fh)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.models))["models"]
    names = args.only if args.only else list(cfg)
    repos = [(n, cfg[n]["hf_id"]) for n in names]

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join("results", f"fetch_{ts}.log")
    fh = open(logpath, "w")
    log(f"prefetch -> {logpath}; cache={hub_cache()}", fh)
    log(f"repos: {[r for _, r in repos]}", fh)

    ok, bad = [], []
    for name, repo in repos:
        (ok if fetch_one(repo, fh) else bad).append(name)

    log("----- FETCH SUMMARY -----", fh)
    log(f"OK : {ok}", fh)
    log(f"BAD: {bad}", fh)
    fh.close()
    print(f"\nlog saved -> {logpath}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
