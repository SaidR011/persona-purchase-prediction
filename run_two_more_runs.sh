#!/bin/bash
# run_two_more_runs.sh - runs 2 and 3 of the main experiment (run 1 = adapters/main_run, seed 0).
# Identical settings to run 1; only the random seed changes.
# Everything printed is ALSO saved to the logs/ folder, so the validation losses aren't lost.
#
# Usage:  caffeinate -i bash run_two_more_runs.sh

set -e            # stop at the first error
set -o pipefail   # ...even when the output is being copied to a log file
mkdir -p logs

for RUN in 2 3; do
  SEED=$((RUN - 1))          # run 2 -> seed 1, run 3 -> seed 2
  echo "========== RUN $RUN (seed $SEED) =========="

  mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --train --data data/training \
    --iters 2000 --batch-size 8 --mask-prompt --learning-rate 1e-4 \
    --steps-per-report 50 --steps-per-eval 250 --val-batches 25 --save-every 250 \
    --seed $SEED --adapter-path adapters/main_run$RUN 2>&1 | tee logs/main_run${RUN}_training.txt

  python run_finetuned_model.py --adapter adapters/main_run$RUN            2>&1 | tee logs/main_run${RUN}_scoring.txt
  python run_finetuned_model.py --adapter adapters/main_run$RUN --stranger 2>&1 | tee logs/main_run${RUN}_stranger.txt
done

echo "========== ALL DONE =========="
