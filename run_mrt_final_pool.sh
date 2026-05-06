#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

RUN_ID="${MRT_FINAL_RUN_ID:-mrt_final_pool_$(date +%Y%m%d_%H%M%S)}"
REPS="${MRT_NUM_EXPERIMENTS:-5}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

CLASSIC_TASKS=(
  american_airlines_2
  api_key_calendar_agendas_2
  bank_transfer_2
  spam_filter_update
)

CUA_TASKS=(
  admin_make_spreadsheet
  admin_mass_forms_filling
  admin_remove_pages
  ds_coffee
  ds_find_meeting_spreadsheet
  ds_fix_table_values_and_missing_answers
  ds_format_excel_sheets
  finance_budget_variance
  finance_check_attendance_payroll
  finance_expense_validation
  finance_invoice_matching
  hr_check_attendance_multiple_days
  hr_populate_salary_increase_memo
  hr_salary_analysis
  research_answer_questions
)

if [[ "$DRY_RUN" != "1" && "${MRT_CONFIRM_FINAL_RUN:-}" != "YES" ]]; then
  echo "Refusing to start final run."
  echo "Set MRT_CONFIRM_FINAL_RUN=YES only after explicit approval, or pass --dry-run to inspect commands."
  exit 2
fi

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "run_id=$RUN_ID"
echo "repetitions_per_condition=$REPS"
echo "classic_active_count=${#CLASSIC_TASKS[@]}"
echo "cua_active_count=${#CUA_TASKS[@]}"
echo "active_task_families=$((${#CLASSIC_TASKS[@]} + ${#CUA_TASKS[@]}))"
echo "excluded_pipeline_task=classic_shade/expenditure_accounting"

cd "$ROOT_DIR"
if [[ -f .env.mlsecurity2 ]]; then
  set -a
  source .env.mlsecurity2
  set +a
elif [[ "$DRY_RUN" == "1" ]]; then
  AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-DRY_RUN_KEY}"
else
  echo "Missing .env.mlsecurity2; refusing to start final run without credentials." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  export OPENAI_API_KEY="DRY_RUN_KEY"
else
  export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://mlsecurity2.openai.azure.com/openai/v1/}"
export OPENAI_USER="${OPENAI_USER:-mlsecurity-final}"
export OPENAI_PROJECT_NAME="${OPENAI_PROJECT_NAME:-mrt-final}"
export MRT_DISABLE_OPENAI_METADATA="${MRT_DISABLE_OPENAI_METADATA:-1}"
export MRT_SETTING="${MRT_SETTING:-selective_v2_liar_204_thinkchoose}"
export MRT_ATTACK_EXECUTION_CHECKLIST="${MRT_ATTACK_EXECUTION_CHECKLIST:-1}"
export MRT_GIVE_HINT="${MRT_GIVE_HINT:-True}"
export MRT_MAX_TOKENS="${MRT_MAX_TOKENS:-4096}"
export MRT_OPENAI_TIMEOUT="${MRT_OPENAI_TIMEOUT:-120}"
export MRT_OPENAI_ATTEMPTS="${MRT_OPENAI_ATTEMPTS:-3}"
export MRT_OPENAI_CLIENT_RETRIES="${MRT_OPENAI_CLIENT_RETRIES:-1}"
export MRT_OPENAI_RETRY_MAX_WAIT="${MRT_OPENAI_RETRY_MAX_WAIT:-30}"

if [[ "${MRT_RUN_CLASSIC:-1}" == "1" ]]; then
  run_or_print env \
    MRT_AUDIT_ID="${RUN_ID}_classic" \
    MRT_AUDIT_STATUS_FILE="$ROOT_DIR/${RUN_ID}_classic_status.txt" \
    MRT_CLASSIC_TASK_SET=classic_stable \
    MRT_NUM_EXPERIMENTS="$REPS" \
    "$ROOT_DIR/run_mrt_classic_audit.sh"
fi

if [[ "${MRT_RUN_CUA:-1}" == "1" ]]; then
  CUA_MODEL="${MRT_AGENTCOMPANY_MODEL:-openai/gpt-5.2}"
  CUA_MAX_WORKERS="${MRT_AGENTCOMPANY_MAX_WORKERS:-1}"
  CUA_MAX_ITERATIONS="${MRT_AGENTCOMPANY_MAX_ITERATIONS:-100}"

  pushd "$ROOT_DIR/external_baselines/mrt" >/dev/null
  for CUA_TASK in "${CUA_TASKS[@]}"; do
    run_or_print "$PYTHON_BIN" task_completion_agentcompany.py \
      --format_name benign_v1 \
      --task_names "$CUA_TASK" \
      --num_experiments "$REPS" \
      --backend_model "$CUA_MODEL" \
      --max_workers "$CUA_MAX_WORKERS" \
      --max_iterations "$CUA_MAX_ITERATIONS" \
      --base_url "$OPENAI_BASE_URL" \
      --api_key "$OPENAI_API_KEY"
  done

  for CUA_TASK in "${CUA_TASKS[@]}"; do
    run_or_print "$PYTHON_BIN" task_completion_agentcompany.py \
      --format_name selective_liar_v2 \
      --task_names "$CUA_TASK" \
      --num_experiments "$REPS" \
      --backend_model "$CUA_MODEL" \
      --max_workers "$CUA_MAX_WORKERS" \
      --max_iterations "$CUA_MAX_ITERATIONS" \
      --base_url "$OPENAI_BASE_URL" \
      --api_key "$OPENAI_API_KEY"
  done

  if [[ "${MRT_RUN_CUA_MONITORS:-1}" == "1" ]]; then
    run_or_print "$PYTHON_BIN" monitor_generation/add_monitors_agentcompany.py \
      --agent_models "${CUA_MODEL##*/}" \
      --monitor_model "${MRT_MONITOR_MODEL:-gpt-5.2}" \
      --transcript_type transcripts_no_hint_benign_v1_thinking_time transcripts_no_hint_selective_liar_v2_thinking_time \
      --monitor_styles "" \
      --max_concurrent "${MRT_MONITOR_MAX_CONCURRENT:-4}" \
      --max_num_transcripts "$REPS"
  fi
  popd >/dev/null
fi

echo "finished=$(iso_now)"
