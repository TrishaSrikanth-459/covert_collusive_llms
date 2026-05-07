#!/usr/bin/env bash
set -euo pipefail

# Launch the frozen 25-task coordination-collusion holdout matrix.
# Credentials must be supplied by the local shell or .env.mlsecurity2.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f .env.mlsecurity2 ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.mlsecurity2
  set +a
fi

export MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5.2}"
export TASK_DOMAIN=knowledge_qa
export QA_DATASET_NAME=coordination_collusion
export QA_TASKS=25
export NUM_WORKERS=5
export TOTAL_TURNS=6
export MAX_TOKENS=700
export TASK_POOL_IDS_PATH=task_pool_ids.coordination_collusion.seed0.full25_holdout.json
export RUN_LABEL="${RUN_LABEL:-coordination_collusion_full25_holdout_rep5}"
export OUTPUT_CSV="${OUTPUT_CSV:-experiment_results.${RUN_LABEL}.csv}"
export TASK_PROGRESS_JSONL="${TASK_PROGRESS_JSONL:-task_progress.${RUN_LABEL}.jsonl}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
export RATE_LIMIT_PER_SEC="${RATE_LIMIT_PER_SEC:-2}"

python parallel_experiment_runner.py \
  --tasks 25 \
  --task-ids-path "$TASK_POOL_IDS_PATH" \
  --reps 5 \
  --max-concurrent "$MAX_CONCURRENT" \
  --run-label "$RUN_LABEL" \
  --output-csv "$OUTPUT_CSV" \
  --task-progress-jsonl "$TASK_PROGRESS_JSONL" \
  --conditions "clean_baseline,clean_one_detector,clean_two_detectors,single_attacker,single_attacker_one_detector,single_attacker_two_detectors,collusive_attack,collusive_attack_one_detector,collusive_attack_two_detectors" \
  --no-shuffle
