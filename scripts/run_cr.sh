#!/usr/bin/env bash
# A1 camera-ready runs: three stages, strictly sequential, unattended.
#
#   tmux new -s a1cr
#   cd ~/Documents/vlm-eval && bash scripts/run_cr.sh
#   Ctrl-b d            # detach; tmux attach -t a1cr to look in
#
# Runs on ONE GPU, one model at a time (subprocess-per-unit), so peak host RAM stays at
# a single model's load and nothing has to be babysat. Each stage is attempted first
# with --no-resume in a fresh directory (a clean pass); if it dies part-way, the next
# attempt resumes rather than restarting, so a crash costs minutes, not hours. A stage
# that still fails does not block the following stages.
#
# Env knobs:
#   GPU=0                    CUDA device index
#   VLM_DATA_ROOT=datasets   where download_a1.py put the data
#   MAX_ATTEMPTS=3           attempts per stage before moving on
#   STAGES="A B C"           subset of stages to run
set -uo pipefail

GPU="${GPU:-0}"
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-datasets}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# --- environment -----------------------------------------------------------------
# The runner needs the project venv (torch, transformers). Activate it explicitly so
# the script works under nohup/tmux/cron where no shell rc has been sourced.
# Override with:  VENV=/path/to/venv bash scripts/run_cr.sh
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for cand in "${VENV:-}" ./venv ./.venv ../venv "$HOME/venv" "$HOME/Documents/vlm-eval/venv"; do
    [ -n "$cand" ] && [ -f "$cand/bin/activate" ] && { . "$cand/bin/activate"; break; }
  done
fi
echo "[cr] python  : $(command -v python)"
echo "[cr] venv    : ${VIRTUAL_ENV:-<none active>}"
if ! python -c "import torch, transformers" 2>/dev/null; then
  echo "[cr] FATAL: this python has no torch/transformers."
  echo "[cr]        Activate the project venv first, e.g.:"
  echo "[cr]            source venv/bin/activate && bash scripts/run_cr.sh"
  echo "[cr]        or point the script at it:  VENV=/path/to/venv bash scripts/run_cr.sh"
  exit 1
fi

MODELS_CFG=configs/models.yaml
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
STAGES="${STAGES:-A B C}"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/cr_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================================="
echo "[cr] start $(date -Is)  gpu=$GPU  data=$VLM_DATA_ROOT  stages='$STAGES'"
echo "[cr] log -> $LOG"
echo "=========================================================================="

# completed units = one .json result file per unit (written only on success)
count_done () { ls "$1"/a1__*.json 2>/dev/null | wc -l | tr -d ' '; }

run_stage () {
  local name="$1" cfg="$2" dir="$3" expected="$4"
  echo ""
  echo "--------------------------------------------------------------------------"
  echo "[cr][$name] config=$cfg  dir=$dir  expected=$expected units  $(date -Is)"
  echo "--------------------------------------------------------------------------"

  local attempt=1 done_n=0
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    local flags="--no-resume"
    if [ "$attempt" -gt 1 ]; then
      flags=""                       # retry: keep finished units, fill the gaps
      echo "[cr][$name] attempt $attempt/$MAX_ATTEMPTS (resuming; $(count_done "$dir")/$expected already done)"
    else
      rm -rf "$dir"                  # attempt 1 is a clean pass into a fresh directory
      echo "[cr][$name] attempt 1/$MAX_ATTEMPTS (clean pass, --no-resume)"
    fi

    python -m runner.orchestrate \
      --run-config "$cfg" \
      --models-config "$MODELS_CFG" \
      --results-dir "$dir" \
      --gpu "$GPU" $flags
    local rc=$?

    done_n=$(count_done "$dir")
    echo "[cr][$name] orchestrator exit=$rc  completed=$done_n/$expected"
    if [ "$done_n" -ge "$expected" ]; then
      echo "[cr][$name] COMPLETE $(date -Is)"
      python -m runner.merge --results "$dir" --paper a1 --out "${dir}.csv" \
        && echo "[cr][$name] merged -> ${dir}.csv"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 10
  done

  echo "[cr][$name] INCOMPLETE after $MAX_ATTEMPTS attempts: $done_n/$expected units"
  python -m runner.merge --results "$dir" --paper a1 --out "${dir}.csv" 2>/dev/null \
    && echo "[cr][$name] partial merge -> ${dir}.csv"
  return 1
}

declare -A STATUS

for s in $STAGES; do
  case "$s" in
    A) run_stage "A-imgctrl"  configs/a1_cr_imgctrl.yaml  results_cr_imgctrl  27
       STATUS[A]=$? ;;
    B) run_stage "B-P4"       configs/a1_cr_P4.yaml       results_cr_P4        9
       STATUS[B4]=$?
       run_stage "B-P5"       configs/a1_cr_P5.yaml       results_cr_P5        9
       STATUS[B5]=$? ;;
    C) run_stage "C-corrupt2" configs/a1_cr_corrupt2.yaml results_cr_corrupt2 108
       STATUS[C]=$? ;;
    *) echo "[cr] unknown stage '$s', skipping" ;;
  esac
done

echo ""
echo "=========================================================================="
echo "[cr] ALL STAGES FINISHED $(date -Is)"
for k in "${!STATUS[@]}"; do
  if [ "${STATUS[$k]}" -eq 0 ]; then echo "  stage $k: OK"; else echo "  stage $k: INCOMPLETE (see log)"; fi
done
echo ""
echo "  results_cr_imgctrl  : $(count_done results_cr_imgctrl)/27   (gray | shuffled | mismatched)"
echo "  results_cr_P4       : $(count_done results_cr_P4)/9        (strict formatting prompt)"
echo "  results_cr_P5       : $(count_done results_cr_P5)/9        (calibrated abstention prompt)"
echo "  results_cr_corrupt2 : $(count_done results_cr_corrupt2)/108 (6 new corruption families)"
echo ""
echo "[cr] next: pack the new outputs and send them over:"
echo "  tar czf a1_cr_results.tgz results_cr_*/a1__*.items.jsonl results_cr_*/a1__*.json \\"
echo "      results_cr_*.csv logs/cr_${STAMP}.log"
echo "=========================================================================="
