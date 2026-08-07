# Evaluation-protocol audit of open VLMs on VizWiz

Code and per-prediction outputs for

> **Do VLMs See What Blind Users Show Them? A Distribution-Shift and
> Evaluation-Protocol Audit on VizWiz**
> Sergei O. Kurashkin, Vadim Tynchenko, Aleksei Borodulin
> ACVR 2026 workshop, ECCV 2026.

Nine open vision-language models (all ≤13B) are evaluated on VizWiz-VQA against VQAv2
under a job-based runner: each **unit** = `(model, dataset, condition, seed)` runs as an
isolated subprocess with per-unit timeout, atomic write, and item-level resume. The
release reproduces every number in the paper from the raw predictions.

## What the paper measures

With models and data held fixed, the reported VizWiz number is governed by three
protocol choices that papers usually leave implicit: the **answerability split**, the
**abstention instruction** in the prompt, and the **answer matcher**. The repository
contains the runs behind each.

## Layout

```
configs/     a1_acvr.yaml            main run (9 models x 2 datasets x 10 conditions)
             a1_psweep_P0..P3.yaml   prompt sweep (default / none / permissive / strict)
             a1_imgq2.yaml           image-vs-question decomposition (no-abstention prompt)
             a1_cr_imgctrl.yaml      image-provenance controls (grey / shuffled / mismatched)
             a1_cr_P4.yaml           strict output-format prompt
             a1_cr_P5.yaml           calibrated-abstention prompt
             a1_cr_corrupt2.yaml     six further corruption families x 2 severities
             models.yaml             the nine checkpoints
harness/     model adapters, VLM facade, VQA parsing, corruption operators
data/        download_a1.py, datasets.py (unified item loaders)
runner/      units.py, worker.py, orchestrate.py (subprocess per unit), merge.py
analysis/    run_a1_split, rescore (strict/lenient/token-F1), compare_prompts,
             decompose_axes, cr_offline, answerability, figures, stats, make_cd
gate/        review_gate.py + gate_config_a1.yaml (fails if dirty numbers reach a figure)
scripts/     run_cr.sh (unattended camera-ready run driver)
results_*/   per-prediction outputs: {id, pred, acc, abstain}; no ground truth
```

## Reproducing

Offline analyses need no GPU and run against the shipped `results_*/`.

```bash
pip install -r requirements.txt
python -m data.download_a1                    # VizWiz-val and VQAv2-val (public)
```

Main run and the frozen numbers:

```bash
python -m runner.orchestrate --run-config configs/a1_acvr.yaml \
    --models-config configs/models.yaml --no-resume --results-dir results_final

python -m analysis.run_a1_split --results results_final --config configs/a1_acvr.yaml \
    --datasets-root datasets --outdir analysis/out_split      # shift, abstention, corruption
python -m analysis.rescore      --results results_final --datasets-root datasets \
    --out analysis/out_split/a1_bracket.json                  # strict / lenient / token-F1
```

Prompt sensitivity, and the image-versus-question decomposition:

```bash
for P in P0 P1 P2 P3; do
  python -m runner.orchestrate --run-config configs/a1_psweep_$P.yaml \
      --models-config configs/models.yaml --no-resume
done
python -m analysis.compare_prompts P0=... P1=... P2=... P3=...     # see analysis/out_psweep_*

python -m runner.orchestrate --run-config configs/a1_imgq2.yaml \
    --models-config configs/models.yaml --no-resume
python -m analysis.decompose_axes --results results_imgq2 --datasets-root datasets
```

Camera-ready additions (image-provenance controls, two further prompts, six further
corruption families) run unattended as three sequential stages:

```bash
bash scripts/run_cr.sh          # ~27 GPU-h on one A800; stages A, B, C in order
python -m analysis.cr_offline --results-final results_final \
    --psweep results_psweep_P0 results_psweep_P1 results_psweep_P2 results_psweep_P3 \
    --datasets-root datasets
```

Before numbers are frozen:

```bash
python gate/review_gate.py results_final --config gate/gate_config_a1.yaml
```

See `REPRODUCIBILITY.md` for the environment pin and the clean-final-pass procedure.

## Per-prediction outputs

The raw per-prediction outputs for every run in the paper are shipped as a single
archive, `results/a1_results.tgz` (unpack it in the repository root before running any
analysis):

```bash
tar xzf results/a1_results.tgz
```

This restores:

```
results_final/        main run: 9 models x 2 datasets x clean / blind / corruption
results_psweep_P0..P3/  prompt sweep (default / none / permissive / strict)
results_imgq2/        image-vs-question decomposition (no-abstention prompt)
results_cr_imgctrl/   image-provenance controls (grey / shuffled / mismatched)
results_cr_P4, _P5/   strict-format and calibrated-abstention prompts
results_cr_corrupt2/  six further corruption families at two severities
```

Each `*.items.jsonl` holds one JSON object per item: `{id, pred, acc, abstain}`. No
reference answers are redistributed; scoring reads the public VizWiz and VQAv2 ground
truth locally via `data/download_a1.py`.


## Notes

- Per-prediction files carry no reference answers; scoring reads the public VizWiz and
  VQAv2 ground truth locally, which is not redistributed here.
- Decoding is greedy (`temperature=0`), 64 new tokens maximum. The exact prompt and the
  matcher normalisation are given in the paper's supplementary material.
- `a1_imgq2.yaml` deliberately uses a no-abstention prompt: under the default prompt an
  imageless model replies "unanswerable", which scores zero on VQAv2 and would corrupt
  the language-prior floor.

## Citation

```bibtex
@inproceedings{kurashkin2026vizwizaudit,
  title     = {Do VLMs See What Blind Users Show Them? A Distribution-Shift and
               Evaluation-Protocol Audit on VizWiz},
  author    = {Kurashkin, Sergei O. and Tynchenko, Vadim and Borodulin, Aleksei},
  booktitle = {ECCV 2026 Workshops (ACVR)},
  year      = {2026}
}
```

Released under the MIT licence.
