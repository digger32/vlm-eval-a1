#!/usr/bin/env bash
# A1 (ACVR) full run. Always inside tmux so it survives SSH disconnect.
#
#   tmux new -s a1            # then:  bash scripts/run_a1.sh
#   Ctrl-b d                  # detach;  tmux attach -t a1 to reattach
#
# Env knobs:
#   GPU=0                     CUDA device index for this run
#   VLM_DATA_ROOT=datasets    where download_a1.py put the data
set -euo pipefail

GPU="${GPU:-0}"
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-datasets}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[run_a1] gpu=$GPU data=$VLM_DATA_ROOT"

python -m runner.orchestrate \
  --run-config configs/a1_acvr.yaml \
  --models-config configs/models.yaml \
  --gpu "$GPU"

python -m runner.merge --results results --paper a1 --out results_a1.csv
echo "[run_a1] done -> results_a1.csv"
