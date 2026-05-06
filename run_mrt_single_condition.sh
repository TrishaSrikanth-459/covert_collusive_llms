#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${MRT_TASK_NAME:?Set MRT_TASK_NAME}"
CONDITION="${MRT_CONDITION:?Set MRT_CONDITION to honest or attack}"
AUDIT_ID="${MRT_AUDIT_ID:-mrt_single_$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="${MRT_STATUS_FILE:-$ROOT_DIR/${AUDIT_ID}_status.txt}"
MAX_TOKENS="${MRT_MAX_TOKENS:-4096}"
LOG_FILE="${MRT_LOG_FILE:-$ROOT_DIR/${AUDIT_ID}_${TASK_NAME}_${CONDITION}.log}"

if [[ "$CONDITION" != "honest" && "$CONDITION" != "attack" ]]; then
  echo "MRT_CONDITION must be honest or attack, got: $CONDITION" >&2
  exit 2
fi

cd "$ROOT_DIR"

set -a
source .env.mlsecurity2
set +a

export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://mlsecurity2.openai.azure.com/openai/v1/}"
export OPENAI_USER="${OPENAI_USER:-mlsecurity-audit}"
export OPENAI_PROJECT_NAME="${OPENAI_PROJECT_NAME:-mrt-audit}"
export MRT_DISABLE_OPENAI_METADATA="${MRT_DISABLE_OPENAI_METADATA:-1}"
export MRT_MAX_TOKENS="$MAX_TOKENS"
export MRT_OPENAI_TIMEOUT="${MRT_OPENAI_TIMEOUT:-120}"
export MRT_OPENAI_ATTEMPTS="${MRT_OPENAI_ATTEMPTS:-3}"
export MRT_OPENAI_CLIENT_RETRIES="${MRT_OPENAI_CLIENT_RETRIES:-1}"
export MRT_OPENAI_RETRY_MAX_WAIT="${MRT_OPENAI_RETRY_MAX_WAIT:-30}"
export MRT_ATTACK_EXECUTION_CHECKLIST="${MRT_ATTACK_EXECUTION_CHECKLIST:-1}"
export MRT_TASK_NAME="$TASK_NAME"
export MRT_NUM_EXPERIMENTS="${MRT_NUM_EXPERIMENTS:-1}"
export MRT_SETTING="${MRT_SETTING:-selective_v2_liar_204_thinkchoose}"
export MRT_GIVE_HINT="${MRT_GIVE_HINT:-True}"
export MRT_MAIN_TASK_INSTRUCTIONS="${MRT_MAIN_TASK_INSTRUCTIONS:-False}"

echo "running=$TASK_NAME/$CONDITION $(date -Is) log=$LOG_FILE" >> "$STATUS_FILE"

if [[ "$CONDITION" == "honest" ]]; then
  MRT_TEST_MODE=False MRT_INCLUDE_SECONDARY_TASK=False ./run_mrt_classic_task.sh > "$LOG_FILE" 2>&1
else
  MRT_TEST_MODE=False MRT_INCLUDE_SECONDARY_TASK=True ./run_mrt_classic_task.sh > "$LOG_FILE" 2>&1
fi

echo "done=$TASK_NAME/$CONDITION $(date -Is) log=$LOG_FILE" >> "$STATUS_FILE"
