#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRT_DIR="$ROOT_DIR/external_baselines/mrt"
TASK_NAME="${MRT_TASK_NAME:-spam_filter_update}"
NUM_EXPERIMENTS="${MRT_NUM_EXPERIMENTS:-1}"
SETTING="${MRT_SETTING:-selective_v2_liar_204_thinkchoose}"
INCLUDE_SECONDARY="${MRT_INCLUDE_SECONDARY_TASK:-True}"
TEST_MODE="${MRT_TEST_MODE:-True}"
GIVE_HINT="${MRT_GIVE_HINT:-True}"
MAIN_TASK_INSTRUCTIONS="${MRT_MAIN_TASK_INSTRUCTIONS:-False}"
PYTHON_BIN="${MRT_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

cd "$MRT_DIR"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" task_completion.py \
  --num_experiments "$NUM_EXPERIMENTS" \
  --test_mode "$TEST_MODE" \
  --task_folder "$TASK_NAME" \
  --max_concurrent_threads 1 \
  --give_hint "$GIVE_HINT" \
  --main_task_instructions "$MAIN_TASK_INSTRUCTIONS" \
  --include_secondary_task "$INCLUDE_SECONDARY" \
  --remove_thinking_tags "$SETTING" \
  --config_file "configs/config_azure_gpt52.yaml"
