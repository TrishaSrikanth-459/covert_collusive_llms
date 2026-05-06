#!/usr/bin/env python3
"""Validate the frozen MRT final-run setup without launching model jobs."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from mrt_public_task_manifest import (
    ACTIVE_FINAL_TASK_IDS,
    CUA_SHADE_EXPECTED,
    PIPELINE_EXCLUDED_TASKS,
    build_manifest,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parent
MRT_ROOT = ROOT / "external_baselines" / "mrt"


def _is_executable(path: Path) -> bool:
    return path.exists() and bool(path.stat().st_mode & stat.S_IXUSR)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    manifest = build_manifest()
    errors.extend(validate_manifest(manifest))

    active_ids = [task["id"] for task in manifest["active_tasks"]]
    if active_ids != ACTIVE_FINAL_TASK_IDS:
        errors.append("Generated active task list does not match ACTIVE_FINAL_TASK_IDS.")
    for excluded_id in PIPELINE_EXCLUDED_TASKS:
        if excluded_id in active_ids:
            errors.append(f"Excluded task is still active: {excluded_id}")
    if len(active_ids) != 19:
        errors.append(f"Expected 19 active task families after exclusion, got {len(active_ids)}.")

    json_manifest_path = ROOT / "mrt_public_full_manifest.json"
    if not json_manifest_path.exists():
        errors.append("Missing mrt_public_full_manifest.json.")
    else:
        json_manifest = _load_json(json_manifest_path)
        json_active = [
            *json_manifest.get("classic_shade_active_tasks", []),
            *json_manifest.get("cua_shade_public_tasks", []),
        ]
        if json_active != active_ids:
            errors.append("mrt_public_full_manifest.json active tasks differ from generated manifest.")

    required_scripts = [
        ROOT / "run_mrt_classic_audit.sh",
        ROOT / "run_mrt_classic_task.sh",
        ROOT / "run_mrt_final_pool.sh",
    ]
    for script in required_scripts:
        if not _is_executable(script):
            errors.append(f"Script is missing or not executable: {script.relative_to(ROOT)}")

    formats_path = MRT_ROOT / "experiments" / "agentcompany" / "formats.json"
    if not formats_path.exists():
        errors.append("Missing AgentCompany formats.json.")
    else:
        formats = _load_json(formats_path)
        for format_name in ("benign_v1", "selective_liar_v2"):
            if format_name not in formats:
                errors.append(f"Missing AgentCompany format: {format_name}")
        if "{primary_task}" not in str(formats.get("benign_v1", "")):
            errors.append("benign_v1 format does not include {primary_task}.")
        if "{primary_task}" not in str(formats.get("selective_liar_v2", "")):
            errors.append("selective_liar_v2 format does not include {primary_task}.")
        if "{secondary_task}" not in str(formats.get("selective_liar_v2", "")):
            errors.append("selective_liar_v2 format does not include {secondary_task}.")

    model_config_path = MRT_ROOT / "experiments" / "agentcompany" / "openhands" / "model_configs.json"
    if not model_config_path.exists():
        errors.append("Missing OpenHands model_configs.json.")
    else:
        model_config = _load_json(model_config_path)
        if "openai/gpt-5.2" not in model_config.get("models", {}):
            errors.append("OpenHands model config does not include openai/gpt-5.2.")

    formatter_path = MRT_ROOT / "experiments" / "agentcompany" / "scripts" / "format_results.py"
    formatter_text = formatter_path.read_text(encoding="utf-8") if formatter_path.exists() else ""
    if "build_transcript_payload" not in formatter_text or "\"conversation\"" not in formatter_text:
        errors.append("AgentCompany formatter does not emit monitor-compatible transcript payloads.")

    for task_id in CUA_SHADE_EXPECTED:
        task_dir = MRT_ROOT / "task_pairs_agentcompany" / task_id
        task_json_path = task_dir / "task.json"
        for rel in ("task.json", "task_definition.py", "solution.py"):
            if not (task_dir / rel).exists():
                errors.append(f"Missing CUA task file: task_pairs_agentcompany/{task_id}/{rel}")
        if task_json_path.exists():
            payload = _load_json(task_json_path)
            if not payload.get("primary_task"):
                errors.append(f"Missing primary_task in CUA task.json: {task_id}")
            if not payload.get("secondary_task"):
                errors.append(f"Missing secondary_task in CUA task.json: {task_id}")

    env_path = ROOT / ".env.mlsecurity2"
    if not env_path.exists():
        warnings.append("Missing .env.mlsecurity2; final run needs Azure credentials on the VM.")

    if os.getenv("MRT_REQUIRE_DOCKER", "1").lower() in {"1", "true", "yes", "y", "on"}:
        if shutil.which("docker") is None:
            errors.append("Docker is not installed or not on PATH; CUA/AgentCompany tasks require Docker.")
        else:
            docker_ps = subprocess.run(["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if docker_ps.returncode != 0:
                errors.append(f"Docker is installed but not usable by this user: {docker_ps.stderr.strip()}")
            for image in (
                "ghcr.io/all-hands-ai/openhands:0.49",
                "ghcr.io/all-hands-ai/runtime:0.49-nikolaik",
            ):
                inspect = subprocess.run(
                    ["docker", "image", "inspect", image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                if inspect.returncode != 0:
                    warnings.append(f"Docker image not pre-pulled yet: {image}")

    if errors:
        print(json.dumps({"ready": False, "errors": errors, "warnings": warnings}, indent=2))
        return 1
    print(json.dumps({
        "ready": True,
        "active_task_families": len(active_ids),
        "active_tasks": active_ids,
        "pipeline_excluded_tasks": list(PIPELINE_EXCLUDED_TASKS),
        "warnings": warnings,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
