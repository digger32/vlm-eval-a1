# Reproducibility (A1)

## Environment (pin on the final pass)
- Python 3.10.12, single NVIDIA A800 80GB (CUDA 13.0)
- torch 2.12.1+cu130, transformers 5.12.1, accelerate 1.14.0
- On the final clean pass run: `pip freeze > results_final/requirements.lock.txt`

## Determinism
- Inference is greedy (`do_sample=false`, `temperature=0`). Each unit
  `(model, dataset, condition, seed)` is an isolated subprocess that loads the model
  fresh and writes its own atomic JSON; **no state is shared between units**, so a
  re-run reproduces identical per-item outputs and there is no resume carry-over by
  construction. `seed` governs only subset sampling.

## Clean-final-pass procedure (the numbers that go in the paper)
```bash
# 1. finish any missing units (resume)
python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
    --models-config configs/models.yaml

# 2. FINAL clean pass: resume OFF, fresh dir (deterministic -> reproduces the same numbers)
python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
    --models-config configs/models.yaml --no-resume --results-dir results_final
pip freeze > results_final/requirements.lock.txt

# 3. gate must PASS before freezing
python gate/review_gate.py results_final --config gate/gate_config_a1.yaml

# 4. analyses on the clean dir -> frozen numbers
python -m runner.merge       --results results_final --paper a1 --out results_a1.csv
python -m analysis.run_a1_split --results results_final --config configs/a1_acvr.yaml --datasets-root datasets --relaxed
python -m analysis.rescore      --results results_final --datasets-root datasets
```

## Data availability
- VizWiz-VQA (val) and VQAv2 (val) are public. Code and per-prediction outputs
  (`results_final/*.items.jsonl`) will be released. State this in the paper; do NOT
  link an identifying repo in the double-blind submission.

## Additional paper analyses (all offline, from the shipped results)
```bash
# token-F1 third matcher + ranking-stability tau (Sec. 4.3, Table 3)
python -m analysis.rescore --results results_final --datasets-root datasets \
    --out analysis/out_split/a1_bracket.json

# prompt-sensitivity sweep (Sec. 4.2): run configs/a1_psweep_P0..P3.yaml, then
python -m analysis.compare_prompts P0=... P1=... P2=... P3=...   # see README

# image-vs-question decomposition (Sec. 4.4): run configs/a1_imgq2.yaml (no-abstention
# prompt, four cells), then
python -m analysis.decompose_axes --results results_imgq2 --datasets-root datasets
```
Note: `a1_imgq2.yaml` uses a no-abstention prompt on purpose; under the default prompt an
imageless model replies "unanswerable", which scores 0 on VQAv2 and corrupts the
language-prior floor (see the paper's Sec. 4.4).
