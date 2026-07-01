#!/usr/bin/env python3
"""
Review-proofing gate.

Reads the runner's outputs (run_meta.json + manifest.jsonl + per-unit JSONs) and
a gate_config.yaml, then asserts the conditions that keep dirty numbers out of
figures. Exits NON-ZERO on any failure so it can block a finalisation step in a
pipeline (e.g. `python review_gate.py runs/final && python make_figures.py`).

Built-in assertions:
  A1  clean final run     final pass had resume DISABLED, no unit skipped
  B1  external validity    every comparative claim has >=1 independent-dataset run
Optional (enable in config):
  C1  calibration present  ECE/coverage recorded for UQ contributions
  D1  optimism gap         inner-CV vs held-out gap recorded for tuned configs
  E1  stats present        omnibus + post-hoc outputs exist for the main comparison

Usage:
    python review_gate.py <outdir> [--config gate_config.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[gate] PyYAML required: pip install pyyaml --break-system-packages")
    sys.exit(2)


def load_manifest(outdir: Path):
    mf = outdir / "manifest.jsonl"
    if not mf.exists():
        return []
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]


def load_units(outdir: Path):
    units = []
    for p in outdir.glob("*__*__*__seed*.json"):
        try:
            units.append(json.loads(p.read_text()))
        except Exception:
            pass
    return units


def check_A1(outdir, manifest, cfg):
    """Final pass had resume DISABLED and skipped nothing."""
    meta_path = outdir / "run_meta.json"
    if not meta_path.exists():
        return False, "run_meta.json missing — cannot verify the final pass"
    meta = json.loads(meta_path.read_text())
    if not meta.get("no_resume", False):
        return False, "final pass ran WITHOUT --no-resume (resume was enabled)"
    if any(r.get("status") == "skip" for r in manifest):
        return False, "manifest shows skipped units in a no-resume pass"
    run_started = meta.get("run_started")
    stale = [r["unit"] for r in manifest if r.get("started") != run_started]
    if stale:
        return False, f"{len(stale)} unit(s) carry a different run_started (carry-over): {stale[:3]}..."
    return True, "final pass clean: --no-resume, no skips, single run_started"


def check_B1(outdir, units, cfg):
    """Every comparative claim has at least one independent-dataset run."""
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared in config — declare them"
    present_datasets = {u.get("dataset") for u in units}
    failures = []
    for c in claims:
        needed = set(c.get("independent_datasets", []))
        if not needed:
            failures.append(f"claim '{c.get('id')}' lists no independent_datasets")
        elif not (needed & present_datasets):
            failures.append(f"claim '{c.get('id')}' has no run on any of {sorted(needed)}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(claims)} comparative claim(s) each have an independent-dataset run"


def check_metric_present(outdir, units, cfg, key, label):
    """Generic: assert some unit recorded a non-null metric `key`."""
    have = [u for u in units
            if (u.get("metrics") or {}).get(key) is not None]
    if not have:
        return False, f"no unit recorded '{key}' ({label})"
    return True, f"'{key}' present in {len(have)} unit(s) ({label})"


def check_E1_stats(outdir, cfg):
    art = cfg.get("stats_artifacts", ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (outdir / a).exists()
               and not Path(a).exists()]
    if missing:
        return False, f"missing stats artifacts: {missing}"
    return True, f"stats artifacts present: {art}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--config", default="gate_config.yaml")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        cfg_path = outdir / a.config
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    manifest = load_manifest(outdir)
    units = load_units(outdir)

    results = []
    results.append(("A1 clean-final-run", *check_A1(outdir, manifest, cfg)))
    results.append(("B1 external-validity", *check_B1(outdir, units, cfg)))
    if cfg.get("require_calibration"):
        results.append(("C1 calibration", *check_metric_present(
            outdir, units, cfg, "ece", "calibration present")))
    if cfg.get("require_optimism_gap"):
        results.append(("D1 optimism-gap", *check_metric_present(
            outdir, units, cfg, "optimism_gap", "HPO honesty")))
    if cfg.get("require_stats"):
        results.append(("E1 stats", *check_E1_stats(outdir, cfg)))

    print("=" * 64)
    print(f"REVIEW-PROOFING GATE  | outdir={outdir}")
    print("=" * 64)
    ok = True
    for name, passed, msg in results:
        flag = "PASS" if passed else "FAIL"
        print(f"[{flag}] {name:24s} {msg}")
        ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
