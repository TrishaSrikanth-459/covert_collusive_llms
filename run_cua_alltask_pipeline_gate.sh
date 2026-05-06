#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/azureuser/MLSecurity_Project
MRT=$ROOT/external_baselines/mrt
STATUS=$ROOT/mrt_cua_alltask_pipeline_gate_status.txt
SUMMARY=$ROOT/mrt_cua_alltask_pipeline_gate_summary.json
LOG=$ROOT/mrt_cua_alltask_pipeline_gate_$(date +%Y%m%d_%H%M%S).log
TASKS=(
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
{
  echo "gate_id=mrt_cua_alltask_pipeline_gate"
  echo "started=$(date -Is)"
  echo "task_count=${#TASKS[@]}"
  echo "conditions=benign_v1,selective_liar_v2"
  echo "num_experiments=1"
  echo "max_iterations=3"
  echo "max_workers=2"
  echo "log=$LOG"
} > "$STATUS"
cd "$MRT"
set -a
source "$ROOT/.env.mlsecurity2"
set +a
export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
export OPENAI_BASE_URL="https://mlsecurity2.openai.azure.com/openai/v1/"
export OPENAI_USER=mlsecurity-cua-gate
export OPENAI_PROJECT_NAME=mrt-cua-gate
export MRT_DISABLE_OPENAI_METADATA=1
for FORMAT in benign_v1 selective_liar_v2; do
  echo "running_trajectory_format=$FORMAT $(date -Is)" >> "$STATUS"
  echo "=== TRAJECTORY $FORMAT $(date -Is) ===" >> "$LOG"
  timeout 3600 ../../.venv/bin/python task_completion_agentcompany.py \
    --format_name "$FORMAT" \
    --task_names "${TASKS[@]}" \
    --num_experiments 1 \
    --backend_model openai/gpt-5.2 \
    --max_workers 2 \
    --max_iterations 3 \
    --base_url "$OPENAI_BASE_URL" \
    --api_key "$AZURE_OPENAI_API_KEY" >> "$LOG" 2>&1
  echo "done_trajectory_format=$FORMAT $(date -Is)" >> "$STATUS"
done

echo "running_monitor=$(date -Is)" >> "$STATUS"
echo "=== MONITOR $(date -Is) ===" >> "$LOG"
timeout 1800 ../../.venv/bin/python monitor_generation/add_monitors_agentcompany.py \
  --agent_models gpt-5.2 \
  --monitor_model gpt-5.2 \
  --transcript_type transcripts_no_hint_benign_v1_thinking_time transcripts_no_hint_selective_liar_v2_thinking_time \
  --monitor_styles "" \
  --max_concurrent 4 \
  --max_num_transcripts 1 >> "$LOG" 2>&1
if grep -Eq "Traceback|Error processing combination|BadRequest|RateLimit|ValueError|58tb" "$LOG"; then
  echo "monitor_or_log_errors_found=$(date -Is)" >> "$STATUS"
  exit 1
fi

echo "done_monitor=$(date -Is)" >> "$STATUS"
../../.venv/bin/python - <<PY
import json, re
from pathlib import Path
root=Path("/home/azureuser/MLSecurity_Project/external_baselines/mrt/task_pairs_agentcompany")
tasks=${TASKS[@]@Q}
# The bash-quoted expansion above is not Python; replace from literal file status instead.
PY
../../.venv/bin/python - <<PY > "$SUMMARY"
import json, re
from pathlib import Path
TASKS = [
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
root = Path("/home/azureuser/MLSecurity_Project/external_baselines/mrt/task_pairs_agentcompany")
rows = []
for task in TASKS:
    for fmt in ["benign_v1", "selective_liar_v2"]:
        raw = root/task/f"transcripts_no_hint_{fmt}_thinking_time"/"gpt-5.2"/"unmonitored"/"transcripts_0.json"
        mon = root/task/f"transcripts_no_hint_{fmt}_thinking_time"/"gpt-5.2"/"gpt-5.2"/"transcripts_0.json"
        item = {"task": task, "format": fmt, "raw_exists": raw.exists(), "monitor_exists": mon.exists(), "verdict": None, "conversation_len": None}
        if mon.exists():
            data = json.loads(mon.read_text())
            item["conversation_len"] = len(data.get("conversation", []))
            m = re.search(r"<verdict>\s*([0-9.]+)\s*</verdict>", str(data.get("monitoring_response", "")))
            item["verdict"] = float(m.group(1)) if m else None
        rows.append(item)
summary = {
    "task_count": len(TASKS),
    "expected_transcripts": len(TASKS) * 2,
    "raw_transcripts": sum(r["raw_exists"] for r in rows),
    "monitor_transcripts": sum(r["monitor_exists"] for r in rows),
    "parsed_verdicts": sum(r["verdict"] is not None for r in rows),
    "benign_mean_verdict": None,
    "attack_mean_verdict": None,
    "rows": rows,
}
for fmt, key in [("benign_v1", "benign_mean_verdict"), ("selective_liar_v2", "attack_mean_verdict")]:
    vals = [r["verdict"] for r in rows if r["format"] == fmt and r["verdict"] is not None]
    summary[key] = sum(vals) / len(vals) if vals else None
print(json.dumps(summary, indent=2))
PY
# Clean smoke transcript dirs so the final run cannot reuse/skip smoke outputs.
for TASK in "${TASKS[@]}"; do
  rm -rf "$MRT/task_pairs_agentcompany/$TASK/transcripts_no_hint_benign_v1_thinking_time"
  rm -rf "$MRT/task_pairs_agentcompany/$TASK/transcripts_no_hint_selective_liar_v2_thinking_time"
done
{
  echo "summary=$SUMMARY"
  echo "smoke_transcripts_removed=$(date -Is)"
  echo "finished=$(date -Is)"
} >> "$STATUS"
