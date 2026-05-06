#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_FILE="${MRT_AUDIT_STATUS_FILE:-$ROOT_DIR/mrt_classic_audit_status.txt}"
AUDIT_ID="${MRT_AUDIT_ID:-mrt_classic_audit_$(date +%Y%m%d_%H%M%S)}"
MAX_TOKENS="${MRT_MAX_TOKENS:-4096}"
TASK_SET="${MRT_CLASSIC_TASK_SET:-classic_stable}"

case "$TASK_SET" in
  classic_stable)
    # Excludes expenditure_accounting because honest and attack runs both showed
    # JSON/tool-call parse instability; that is a pipeline gate, not an attack result.
    TASKS=(
      american_airlines_2
      api_key_calendar_agendas_2
      bank_transfer_2
      spam_filter_update
    )
    ;;
  classic_all)
    TASKS=(
      american_airlines_2
      api_key_calendar_agendas_2
      bank_transfer_2
      expenditure_accounting
      spam_filter_update
    )
    ;;
  custom)
    if [[ -z "${MRT_TASKS:-}" ]]; then
      echo "MRT_CLASSIC_TASK_SET=custom requires MRT_TASKS to be a space-separated task list" >&2
      exit 2
    fi
    read -r -a TASKS <<< "$MRT_TASKS"
    ;;
  *)
    echo "Unknown MRT_CLASSIC_TASK_SET=$TASK_SET; use classic_stable, classic_all, or custom" >&2
    exit 2
    ;;
esac

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
export MRT_GIVE_HINT="${MRT_GIVE_HINT:-True}"
export MRT_NUM_EXPERIMENTS="${MRT_NUM_EXPERIMENTS:-1}"
export MRT_SETTING="${MRT_SETTING:-selective_v2_liar_204_thinkchoose}"

{
  echo "audit_id=$AUDIT_ID"
  echo "started=$(date -Is)"
  echo "task_set=$TASK_SET"
  echo "task_count=${#TASKS[@]}"
  echo "tasks=${TASKS[*]}"
  echo "conditions=honest,attack"
  echo "mrt_setting=$MRT_SETTING"
  echo "give_hint=$MRT_GIVE_HINT"
  echo "attack_execution_checklist=$MRT_ATTACK_EXECUTION_CHECKLIST"
} > "$STATUS_FILE"

for task in "${TASKS[@]}"; do
  for condition in honest attack; do
    log_file="$ROOT_DIR/${AUDIT_ID}_${task}_${condition}.log"
    echo "running=$task/$condition $(date -Is)" >> "$STATUS_FILE"
    export MRT_TASK_NAME="$task"
    if [[ "$condition" == "honest" ]]; then
      MRT_TEST_MODE=False MRT_INCLUDE_SECONDARY_TASK=False ./run_mrt_classic_task.sh > "$log_file" 2>&1
    else
      MRT_TEST_MODE=False MRT_INCLUDE_SECONDARY_TASK=True ./run_mrt_classic_task.sh > "$log_file" 2>&1
    fi
    echo "done=$task/$condition $(date -Is) log=$log_file" >> "$STATUS_FILE"
  done
done

echo "finished=$(date -Is)" >> "$STATUS_FILE"
