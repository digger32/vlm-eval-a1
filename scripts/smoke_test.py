#!/usr/bin/env python
"""PHASE-0 GATE. Load every model once, run one image ask() + one text-only ask(), print
and LOG PASS/FAIL per model with full tracebacks.

Writes results/smoke_<timestamp>.log (env header + per-model output + tracebacks) so the
log can be uploaded for debugging. A model that FAILs is dropped (drop the MODEL, not the
experiment). Run prefetch first so this step is offline and only tests adapters:

  python -m scripts.fetch_models --models configs/models.yaml      # robust download
  tmux new -s smoke
  python -m scripts.smoke_test  --models configs/models.yaml       # adapter gate
  python -m scripts.smoke_test  --models configs/models.yaml --only molmo7b_d
"""
from __future__ import annotations
import argparse
import io
import os
import platform
import traceback
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image


def env_header() -> str:
    lines = [f"smoke @ {datetime.now().isoformat(timespec='seconds')}",
             f"python  {platform.python_version()}  ({platform.platform()})"]
    try:
        import torch
        lines.append(f"torch   {torch.__version__}  cuda={torch.version.cuda} "
                     f"avail={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"gpu     {torch.cuda.get_device_name(0)}  "
                         f"free={free/1e9:.1f}/{total/1e9:.1f} GB")
    except Exception as e:
        lines.append(f"torch   <import failed: {e!r}>")
    for mod in ("transformers", "accelerate", "timm", "einops"):
        try:
            m = __import__(mod)
            lines.append(f"{mod:12s} {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"{mod:12s} <not installed>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    from harness.registry import load_models, load_generation_defaults
    from harness.vlm import VLM

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = args.log or os.path.join("results", f"smoke_{ts}.log")
    logf = open(logpath, "w")

    def emit(s=""):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    emit(env_header())
    emit("=" * 60)

    specs = load_models(args.models)
    gdef = load_generation_defaults(args.models)
    img = Image.new("RGB", (336, 336), (127, 127, 127))
    img.paste((255, 0, 0), (50, 50, 150, 150))   # red square to ask about

    names = [args.only] if args.only else list(specs)
    results = {}
    for name in names:
        emit(f"\n=== {name} ({specs[name].hf_id}) ===")
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                vlm = VLM(specs[name], gdef, item_timeout_s=120)
                a = vlm.ask(img, "What color is the square? Answer in one word.")
                b = vlm.ask(None, "Reply with the single word: ok")
                peak = vlm.peak_vram_bytes() / 1e9
            emit(f"  image ask -> {a!r}")
            emit(f"  blind ask -> {b!r}")
            emit(f"  peak VRAM -> {peak:.2f} GB")
            results[name] = "PASS"
            del vlm
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            results[name] = "FAIL"
            emit("  FAILED — traceback:")
            emit(traceback.format_exc())
        captured = buf.getvalue().strip()
        if captured:
            emit("  --- captured stdout/stderr (tail) ---")
            emit("\n".join(captured.splitlines()[-25:]))

    emit("\n" + "=" * 60)
    emit("SMOKE SUMMARY:")
    for n, r in results.items():
        emit(f"  {r}  {n}")
    fails = [n for n, r in results.items() if r == "FAIL"]
    if fails:
        emit(f"\nDrop or fix before the real run: {fails}")
    emit(f"\nlog saved -> {logpath}")
    logf.close()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
