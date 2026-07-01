# Evaluation-protocol audit of open VLMs on VizWiz (A1)

Anonymised code and per-prediction outputs for the ACVR paper *"Do VLMs See What Blind
Users Show Them? A Distribution-Shift and Evaluation-Protocol Audit on VizWiz."*
Provided for double-blind review; it reproduces every number in the paper from the raw
predictions.

Nine open vision-language models (all <=13B) are evaluated on VizWiz-VQA against VQAv2
under a job-based runner: each **unit** = `(model, dataset, condition, seed)` runs as an
isolated subprocess with per-unit timeout, atomic write, and item-level resume.

## Layout
```
configs/    a1_acvr.yaml (main run) · a1_psweep_P0..P3.yaml (prompt sweep)
            a1_imgq2.yaml (image-vs-question control) · models.yaml
harness/    model adapters, VLM facade, MCQ/VQA parsing, corruption operators
data/       download_a1.py · datasets.py (unified item loaders)
runner/     units.py · worker.py · orchestrate.py (subprocess-per-unit)
analysis/   run_a1_split · rescore (strict/lenient/token-F1) · compare_prompts
            · decompose_axes · answerability · figures · stats · make_cd
gate/       review_gate.py + gate_config_a1.yaml (fails on dirty numbers)
results_*/  per-prediction outputs ({id, pred, acc, abstain}; no ground truth)
```

## Reproduce the paper (offline analyses need no GPU)
```bash
pip install -r requirements.txt
python -m data.download_a1                     # VizWiz-val + VQAv2-val (public)

# main run (GPU; tmux) — or use the shipped results_final/
python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
    --models-config configs/models.yaml --no-resume --results-dir results_final

# frozen numbers (offline, from results_final/):
python -m analysis.run_a1_split --results results_final --config configs/a1_acvr.yaml \
    --datasets-root datasets --outdir analysis/out_split     # shift, abstention, corruption
python -m analysis.rescore      --results results_final --datasets-root datasets \
    --out analysis/out_split/a1_bracket.json                 # strict/lenient/token-F1 + tau

# prompt sensitivity (Sec. 4.2): 4 prompts on a fixed 1500-subset
for P in P0 P1 P2 P3; do
  python -m runner.orchestrate --run-config configs/a1_psweep_$P.yaml \
      --models-config configs/models.yaml --no-resume
  python -m analysis.run_a1_split --results results_psweep_$P \
      --config configs/a1_psweep_$P.yaml --datasets-root datasets \
      --outdir analysis/out_psweep_$P
done
python -m analysis.compare_prompts P0=analysis/out_psweep_P0/a1_split_stats.json \
  P1=analysis/out_psweep_P1/a1_split_stats.json \
  P2=analysis/out_psweep_P2/a1_split_stats.json \
  P3=analysis/out_psweep_P3/a1_split_stats.json

# image-vs-question decomposition (Sec. 4.4): four cells under a no-abstention prompt
python -m runner.orchestrate --run-config configs/a1_imgq2.yaml \
    --models-config configs/models.yaml --no-resume        # -> results_imgq2/
python -m analysis.decompose_axes --results results_imgq2 --datasets-root datasets
```
`gate/review_gate.py results_final --config gate/gate_config_a1.yaml` must pass before
numbers are frozen (asserts a no-resume final pass and a public-dataset robustness run).

See `REPRODUCIBILITY.md` for the environment pin and the clean-final-pass procedure.

## Notes
- Per-prediction files contain no reference answers; scoring reads public VizWiz/VQAv2
  ground truth locally (not redistributed here). Download via `data/download_a1.py`.
- Decoding is greedy (`temperature=0`), max 64 new tokens; the exact prompt and the
  matcher normalisation are documented in the paper's reproducibility appendix.
