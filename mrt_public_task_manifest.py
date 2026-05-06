"""Inventory the MRT/SHADE task universe used for the final study.

The public runnable baseline is intentionally frozen from the existing
ScaleAI/mrt release:

* 5 classic SHADE-Arena tool-use task pairs.
* 15 CUA-SHADE-Arena computer-use task pairs.

The original SHADE paper describes 17 classic task pairs, but ControlArena's
public SHADE wrapper documents that only 5 classic task pairs are public; the
remaining 12 are reserved. The active final pool uses every public MRT task
family except task families excluded by a predeclared pipeline-quality gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
MRT_ROOT = REPO_ROOT / "external_baselines" / "mrt"

CLASSIC_SHADE_PUBLIC_EXPECTED = [
    "american_airlines_2",
    "api_key_calendar_agendas_2",
    "bank_transfer_2",
    "expenditure_accounting",
    "spam_filter_update",
]

PIPELINE_EXCLUDED_TASKS = {
    "classic_shade/expenditure_accounting": (
        "Excluded before the final run because repeated honest and attack audits "
        "completed with malformed JSON tool-call parse errors on long file writes. "
        "This is a pipeline-quality exclusion, not an attack-success exclusion."
    ),
}

CLASSIC_SHADE_ACTIVE = [
    task_id
    for task_id in CLASSIC_SHADE_PUBLIC_EXPECTED
    if f"classic_shade/{task_id}" not in PIPELINE_EXCLUDED_TASKS
]

CUA_SHADE_EXPECTED = [
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

PUBLIC_TASK_IDS = [
    *(f"classic_shade/{task_id}" for task_id in CLASSIC_SHADE_PUBLIC_EXPECTED),
    *(f"cua_shade/{task_id}" for task_id in CUA_SHADE_EXPECTED),
]

ACTIVE_FINAL_TASK_IDS = [
    *(f"classic_shade/{task_id}" for task_id in CLASSIC_SHADE_ACTIVE),
    *(f"cua_shade/{task_id}" for task_id in CUA_SHADE_EXPECTED),
]

DEFAULT_FULL_RUN_DESIGN = {
    "active_task_families": len(ACTIVE_FINAL_TASK_IDS),
    "repetitions_per_condition": 5,
    "frozen_attack_strategy": "selective_v2_liar_204_thinkchoose",
    "trajectory_conditions": [
        "honest_baseline",
        "single_attacker",
    ],
    "monitoring": "posthoc monitor scoring on honest and attack trajectories",
    "planned_primary_trajectories": len(ACTIVE_FINAL_TASK_IDS) * 5 * 2,
    "note": (
        "MRT/SHADE is a single-agent hidden-side-task benchmark. Treat this as "
        "an auxiliary baseline, not the collusion study. The primary study must "
        "include the separate multi-agent collusion matrix, where malicious "
        "workers have no private channel and detectors may coordinate privately "
        "with other detectors. Exclude task families only for predeclared "
        "pipeline failures, not because the attack failed."
    ),
}


def _classic_task_summary(task_id: str) -> Dict[str, Any]:
    task_dir = MRT_ROOT / "task_pairs" / task_id
    readme = task_dir / "README.md"
    return {
        "id": f"classic_shade/{task_id}",
        "suite": "classic_shade_public",
        "task_id": task_id,
        "path": str(task_dir.relative_to(REPO_ROOT)),
        "has_task_definition": (task_dir / "task_definition.py").exists(),
        "has_solution": (task_dir / "solution.py").exists(),
        "readme_excerpt": readme.read_text(encoding="utf-8").strip() if readme.exists() else "",
    }


def _cua_task_summary(task_id: str) -> Dict[str, Any]:
    task_dir = MRT_ROOT / "task_pairs_agentcompany" / task_id
    task_json = task_dir / "task.json"
    payload: Dict[str, Any] = {}
    if task_json.exists():
        payload = json.loads(task_json.read_text(encoding="utf-8"))
    return {
        "id": f"cua_shade/{task_id}",
        "suite": "cua_shade_public",
        "task_id": task_id,
        "path": str(task_dir.relative_to(REPO_ROOT)),
        "has_task_json": task_json.exists(),
        "has_task_definition": (task_dir / "task_definition.py").exists(),
        "has_solution": (task_dir / "solution.py").exists(),
        "primary_task": payload.get("primary_task", ""),
        "secondary_task": payload.get("secondary_task", ""),
    }


def build_manifest() -> Dict[str, Any]:
    public_classic_tasks = [_classic_task_summary(task_id) for task_id in CLASSIC_SHADE_PUBLIC_EXPECTED]
    active_classic_tasks = [_classic_task_summary(task_id) for task_id in CLASSIC_SHADE_ACTIVE]
    cua_tasks = [_cua_task_summary(task_id) for task_id in CUA_SHADE_EXPECTED]
    manifest = {
        "baseline": "ScaleAI/mrt public SHADE/CUA-SHADE release",
        "manifest_version": "mrt_public_pipeline_stable_v2",
        "classic_shade_public_task_count": len(public_classic_tasks),
        "classic_shade_active_task_count": len(active_classic_tasks),
        "cua_shade_public_task_count": len(cua_tasks),
        "total_public_task_families": len(public_classic_tasks) + len(cua_tasks),
        "active_final_task_families": len(active_classic_tasks) + len(cua_tasks),
        "pipeline_excluded_tasks": [
            {"id": task_id, "reason": reason}
            for task_id, reason in sorted(PIPELINE_EXCLUDED_TASKS.items())
        ],
        "default_full_run_design": DEFAULT_FULL_RUN_DESIGN,
        "public_tasks": [*public_classic_tasks, *cua_tasks],
        "active_tasks": [*active_classic_tasks, *cua_tasks],
    }
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    public_tasks = manifest.get("public_tasks") or []
    active_tasks = manifest.get("active_tasks") or []
    public_ids = [task.get("id") for task in public_tasks]
    active_ids = [task.get("id") for task in active_tasks]
    if public_ids != PUBLIC_TASK_IDS:
        errors.append("Public task ordering or membership does not match the MRT public pool.")
    if active_ids != ACTIVE_FINAL_TASK_IDS:
        errors.append("Active task ordering or membership does not match the frozen final pool.")
    for excluded_id in PIPELINE_EXCLUDED_TASKS:
        if excluded_id not in public_ids:
            errors.append(f"Pipeline-excluded task is not in public pool: {excluded_id}")
        if excluded_id in active_ids:
            errors.append(f"Pipeline-excluded task is still active: {excluded_id}")
    for task in public_tasks:
        if not task.get("has_task_definition"):
            errors.append(f"Missing task_definition.py for {task.get('id')}")
        if not task.get("has_solution"):
            errors.append(f"Missing solution.py for {task.get('id')}")
        if task.get("suite") == "cua_shade_public" and not task.get("has_task_json"):
            errors.append(f"Missing task.json for {task.get('id')}")
    return errors


def main() -> int:
    manifest = build_manifest()
    errors = validate_manifest(manifest)
    print(json.dumps({
        "manifest_version": manifest["manifest_version"],
        "classic_shade_public_task_count": manifest["classic_shade_public_task_count"],
        "classic_shade_active_task_count": manifest["classic_shade_active_task_count"],
        "cua_shade_public_task_count": manifest["cua_shade_public_task_count"],
        "total_public_task_families": manifest["total_public_task_families"],
        "active_final_task_families": manifest["active_final_task_families"],
        "pipeline_excluded_tasks": manifest["pipeline_excluded_tasks"],
        "default_full_run_design": manifest["default_full_run_design"],
        "validation_errors": errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
