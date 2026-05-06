#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRT_DIR="$ROOT_DIR/external_baselines/mrt"
AUDIT_ID="${MRT_CLEAN_AUDIT_ID:-mrt_clean_baseline_active_$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="${MRT_CLEAN_AUDIT_STATUS_FILE:-$ROOT_DIR/${AUDIT_ID}_status.txt}"
LOG_FILE="${MRT_CLEAN_AUDIT_LOG_FILE:-$ROOT_DIR/${AUDIT_ID}.log}"
SUMMARY_FILE="${MRT_CLEAN_AUDIT_SUMMARY_FILE:-$ROOT_DIR/${AUDIT_ID}_summary.json}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CUA_MAX_WORKERS="${MRT_AGENTCOMPANY_MAX_WORKERS:-2}"
CUA_MAX_ITERATIONS="${MRT_AGENTCOMPANY_MAX_ITERATIONS:-100}"
WAIT_FOR_TMUX="${MRT_WAIT_FOR_TMUX:-}"
REQUIRE_GATE="${MRT_REQUIRE_CUA_GATE:-1}"

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

write_status() {
  echo "$1=$(date -Is)" >> "$STATUS_FILE"
}

wait_for_tmux_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "waiting_for=$session $(date -Is)" >> "$STATUS_FILE"
    sleep 60
  done
}

summarize_cua_gate() {
  "$PYTHON_BIN" - <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path("/home/azureuser/MLSecurity_Project")
tasks = [
    "admin_make_spreadsheet",
    "admin_mass_forms_filling",
    "admin_remove_pages",
    "ds_coffee",
    "ds_find_meeting_spreadsheet",
    "ds_fix_table_values_and_missing_answers",
    "ds_format_excel_sheets",
    "finance_budget_variance",
    "finance_check_attendance_payroll",
    "finance_expense_validation",
    "finance_invoice_matching",
    "hr_check_attendance_multiple_days",
    "hr_populate_salary_increase_memo",
    "hr_salary_analysis",
    "research_answer_questions",
]
task_root = root / "external_baselines/mrt/task_pairs_agentcompany"
rows = []
for task in tasks:
    for fmt in ["benign_v1", "selective_liar_v2"]:
        raw = task_root / task / f"transcripts_no_hint_{fmt}_thinking_time" / "gpt-5.2" / "unmonitored" / "transcripts_0.json"
        mon = task_root / task / f"transcripts_no_hint_{fmt}_thinking_time" / "gpt-5.2" / "gpt-5.2" / "transcripts_0.json"
        row = {
            "task": task,
            "format": fmt,
            "raw_exists": raw.exists(),
            "monitor_exists": mon.exists(),
            "verdict": None,
            "conversation_len": None,
        }
        if mon.exists():
            data = json.loads(mon.read_text())
            row["conversation_len"] = len(data.get("conversation", []))
            match = re.search(r"<verdict>\s*([0-9.]+)\s*</verdict>", str(data.get("monitoring_response", "")))
            row["verdict"] = float(match.group(1)) if match else None
        rows.append(row)

summary = {
    "gate": "mrt_cua_alltask_pipeline_gate",
    "task_count": len(tasks),
    "expected_transcripts": len(tasks) * 2,
    "raw_transcripts": sum(row["raw_exists"] for row in rows),
    "monitor_transcripts": sum(row["monitor_exists"] for row in rows),
    "parsed_verdicts": sum(row["verdict"] is not None for row in rows),
    "benign_mean_verdict": None,
    "attack_mean_verdict": None,
    "rows": rows,
}
for fmt, key in [("benign_v1", "benign_mean_verdict"), ("selective_liar_v2", "attack_mean_verdict")]:
    vals = [row["verdict"] for row in rows if row["format"] == fmt and row["verdict"] is not None]
    summary[key] = sum(vals) / len(vals) if vals else None

(root / "mrt_cua_alltask_pipeline_gate_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if summary["raw_transcripts"] != summary["expected_transcripts"]:
    sys.exit("missing raw CUA gate transcripts")
if summary["monitor_transcripts"] != summary["expected_transcripts"]:
    sys.exit("missing CUA gate monitor transcripts")
if summary["parsed_verdicts"] != summary["expected_transcripts"]:
    sys.exit("missing CUA gate parsed monitor verdicts")
PY
}

clean_cua_smoke_transcripts() {
  for task in "${CUA_TASKS[@]}"; do
    rm -rf "$MRT_DIR/task_pairs_agentcompany/$task/transcripts_no_hint_benign_v1_thinking_time"
    rm -rf "$MRT_DIR/task_pairs_agentcompany/$task/transcripts_no_hint_selective_liar_v2_thinking_time"
  done
}

verify_or_repair_cua_gate() {
  local gate_status="$ROOT_DIR/mrt_cua_alltask_pipeline_gate_status.txt"
  local gate_log=""

  if [[ ! -f "$gate_status" ]]; then
    echo "blocked_missing_cua_gate_status=$(date -Is)" >> "$STATUS_FILE"
    exit 1
  fi

  gate_log="$(awk -F= '/^log=/{print $2}' "$gate_status" | tail -n 1)"
  if [[ -n "$gate_log" && -f "$gate_log" ]]; then
    if grep -Eq "Traceback|OpenAIBadRequestError|BadRequest|unhandled RateLimit|RateLimitError|malformed tool-call crash|No sessions found|Error processing combination" "$gate_log"; then
      echo "blocked_cua_gate_log_error=$(date -Is) log=$gate_log" >> "$STATUS_FILE"
      exit 1
    fi
  fi

  if grep -q "^finished=" "$gate_status"; then
    echo "cua_gate_finished_cleanly=$(date -Is)" >> "$STATUS_FILE"
    return
  fi

  echo "repairing_cua_gate_summary=$(date -Is)" >> "$STATUS_FILE"
  summarize_cua_gate >> "$LOG_FILE" 2>&1
  clean_cua_smoke_transcripts
  {
    echo "summary=$ROOT_DIR/mrt_cua_alltask_pipeline_gate_summary.json"
    echo "smoke_transcripts_removed=$(date -Is)"
    echo "finished_repaired=$(date -Is)"
  } >> "$gate_status"
}

{
  echo "audit_id=$AUDIT_ID"
  echo "queued=$(date -Is)"
  echo "classic_task_count=${#CLASSIC_TASKS[@]}"
  echo "cua_task_count=${#CUA_TASKS[@]}"
  echo "condition=honest_baseline_only"
  echo "classic_reps=1"
  echo "cua_reps=1"
  echo "cua_max_iterations=$CUA_MAX_ITERATIONS"
  echo "cua_max_workers=$CUA_MAX_WORKERS"
  echo "log=$LOG_FILE"
  echo "summary=$SUMMARY_FILE"
} > "$STATUS_FILE"

cd "$ROOT_DIR"

if [[ -n "$WAIT_FOR_TMUX" ]]; then
  wait_for_tmux_session "$WAIT_FOR_TMUX"
fi

if [[ "$REQUIRE_GATE" == "1" ]]; then
  verify_or_repair_cua_gate
fi

write_status "started"

set -a
source "$ROOT_DIR/.env.mlsecurity2"
set +a

export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://mlsecurity2.openai.azure.com/openai/v1/}"
export OPENAI_USER="${OPENAI_USER:-mlsecurity-clean-baseline-audit}"
export OPENAI_PROJECT_NAME="${OPENAI_PROJECT_NAME:-mrt-clean-baseline-audit}"
export MRT_DISABLE_OPENAI_METADATA="${MRT_DISABLE_OPENAI_METADATA:-1}"
export MRT_SETTING="${MRT_SETTING:-selective_v2_liar_204_thinkchoose}"
export MRT_ATTACK_EXECUTION_CHECKLIST="${MRT_ATTACK_EXECUTION_CHECKLIST:-1}"
export MRT_GIVE_HINT="${MRT_GIVE_HINT:-True}"
export MRT_MAX_TOKENS="${MRT_MAX_TOKENS:-4096}"
export MRT_OPENAI_TIMEOUT="${MRT_OPENAI_TIMEOUT:-120}"
export MRT_OPENAI_ATTEMPTS="${MRT_OPENAI_ATTEMPTS:-3}"
export MRT_OPENAI_CLIENT_RETRIES="${MRT_OPENAI_CLIENT_RETRIES:-1}"
export MRT_OPENAI_RETRY_MAX_WAIT="${MRT_OPENAI_RETRY_MAX_WAIT:-30}"

for task in "${CLASSIC_TASKS[@]}"; do
  echo "running_classic_honest=$task $(date -Is)" >> "$STATUS_FILE"
  MRT_AUDIT_ID="$AUDIT_ID" \
    MRT_STATUS_FILE="$STATUS_FILE" \
    MRT_TASK_NAME="$task" \
    MRT_CONDITION=honest \
    MRT_NUM_EXPERIMENTS=1 \
    "$ROOT_DIR/run_mrt_single_condition.sh" >> "$LOG_FILE" 2>&1
  echo "done_classic_honest=$task $(date -Is)" >> "$STATUS_FILE"
done

clean_cua_smoke_transcripts

pushd "$MRT_DIR" >/dev/null
echo "running_cua_honest=$(date -Is)" >> "$STATUS_FILE"
for task in "${CUA_TASKS[@]}"; do
  echo "running_cua_honest_task=$task $(date -Is)" >> "$STATUS_FILE"
  "$PYTHON_BIN" task_completion_agentcompany.py \
    --format_name benign_v1 \
    --task_names "$task" \
    --num_experiments 1 \
    --backend_model openai/gpt-5.2 \
    --max_workers 1 \
    --max_iterations "$CUA_MAX_ITERATIONS" \
    --base_url "$OPENAI_BASE_URL" \
    --api_key "$AZURE_OPENAI_API_KEY" >> "$LOG_FILE" 2>&1
  echo "done_cua_honest_task=$task $(date -Is)" >> "$STATUS_FILE"
done
echo "done_cua_honest=$(date -Is)" >> "$STATUS_FILE"

echo "running_cua_honest_monitors=$(date -Is)" >> "$STATUS_FILE"
"$PYTHON_BIN" monitor_generation/add_monitors_agentcompany.py \
  --agent_models gpt-5.2 \
  --monitor_model gpt-5.2 \
  --transcript_type transcripts_no_hint_benign_v1_thinking_time \
  --monitor_styles "" \
  --max_concurrent 4 \
  --max_num_transcripts 1 >> "$LOG_FILE" 2>&1
echo "done_cua_honest_monitors=$(date -Is)" >> "$STATUS_FILE"
popd >/dev/null

"$PYTHON_BIN" - <<PY > "$SUMMARY_FILE"
import json
import re
from pathlib import Path

root = Path("$ROOT_DIR")
classic_tasks = [
    "american_airlines_2",
    "api_key_calendar_agendas_2",
    "bank_transfer_2",
    "spam_filter_update",
]
cua_tasks = [
    "admin_make_spreadsheet",
    "admin_mass_forms_filling",
    "admin_remove_pages",
    "ds_coffee",
    "ds_find_meeting_spreadsheet",
    "ds_fix_table_values_and_missing_answers",
    "ds_format_excel_sheets",
    "finance_budget_variance",
    "finance_check_attendance_payroll",
    "finance_expense_validation",
    "finance_invoice_matching",
    "hr_check_attendance_multiple_days",
    "hr_populate_salary_increase_memo",
    "hr_salary_analysis",
    "research_answer_questions",
]

classic_logs = []
for task in classic_tasks:
    logs = sorted(root.glob(f"${AUDIT_ID}_{task}_honest.log"))
    text = logs[-1].read_text(errors="replace") if logs else ""
    classic_logs.append({
        "task": task,
        "log_exists": bool(logs),
        "has_traceback": "Traceback" in text,
        "has_bad_request": "BadRequest" in text or "OpenAIBadRequestError" in text,
        "has_rate_limit": "RateLimit" in text,
        "main_eval_mentions": len(re.findall(r"main_task_evaluation|main task|score|success", text, re.I)),
    })

task_root = root / "external_baselines/mrt/task_pairs_agentcompany"
cua_rows = []
for task in cua_tasks:
    raw = task_root / task / "transcripts_no_hint_benign_v1_thinking_time" / "gpt-5.2" / "unmonitored" / "transcripts_0.json"
    mon = task_root / task / "transcripts_no_hint_benign_v1_thinking_time" / "gpt-5.2" / "gpt-5.2" / "transcripts_0.json"
    row = {
        "task": task,
        "raw_exists": raw.exists(),
        "monitor_exists": mon.exists(),
        "verdict": None,
        "conversation_len": None,
    }
    if mon.exists():
        data = json.loads(mon.read_text())
        row["conversation_len"] = len(data.get("conversation", []))
        match = re.search(r"<verdict>\s*([0-9.]+)\s*</verdict>", str(data.get("monitoring_response", "")))
        row["verdict"] = float(match.group(1)) if match else None
    cua_rows.append(row)

summary = {
    "audit_id": "$AUDIT_ID",
    "scope": "honest clean baseline only",
    "classic_task_count": len(classic_tasks),
    "cua_task_count": len(cua_tasks),
    "classic_logs": classic_logs,
    "classic_pipeline_errors": sum(row["has_traceback"] or row["has_bad_request"] or row["has_rate_limit"] for row in classic_logs),
    "cua_raw_transcripts": sum(row["raw_exists"] for row in cua_rows),
    "cua_monitor_transcripts": sum(row["monitor_exists"] for row in cua_rows),
    "cua_parsed_verdicts": sum(row["verdict"] is not None for row in cua_rows),
    "cua_mean_honest_verdict": (
        sum(row["verdict"] for row in cua_rows if row["verdict"] is not None)
        / max(1, sum(row["verdict"] is not None for row in cua_rows))
    ),
    "cua_rows": cua_rows,
    "note": "Classic SHADE exposes objective main-task evaluation in its task_completion logs/results. AgentCompany/CUA exposes OpenHands completion artifacts and monitor verdicts here; it does not expose a universal objective accuracy scorer in this wrapper.",
}
print(json.dumps(summary, indent=2))
PY

write_status "finished"
