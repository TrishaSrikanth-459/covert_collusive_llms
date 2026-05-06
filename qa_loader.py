"""
Dataset-dispatch loader for the active Knowledge-QA experiment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from bigcodebench_loader import load_bigcodebench_tasks
import config
from apps_control_loader import load_apps_control_tasks
from coordination_collusion_loader import load_coordination_collusion_tasks
from decision_sabotage_loader import load_decision_sabotage_tasks
from humaneval_loader import load_humaneval_tasks
from mmlu_pro_loader import load_mmlu_pro_tasks
from musique_loader import load_musique_tasks
from musr_loader import load_musr_tasks
from qasc_loader import load_qasc_tasks


def load_qa_tasks(
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dataset_name = config.QA_DATASET_NAME
    if dataset_name == "musique":
        return load_musique_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name == "qasc":
        return load_qasc_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name == "mmlu_pro":
        return load_mmlu_pro_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name == "musr":
        return load_musr_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name == "decision_sabotage":
        return load_decision_sabotage_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name == "coordination_collusion":
        return load_coordination_collusion_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if dataset_name in {"apps_control", "apps_control_review"}:
        return load_apps_control_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    raise ValueError(f"Unsupported QA_DATASET_NAME: {dataset_name}")


def load_domain_tasks(
    domain: str,
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if domain == "knowledge_qa":
        return load_qa_tasks(
            num_tasks=num_tasks,
            seed=seed,
            start_index=start_index,
            task_ids=task_ids,
            task_ids_path=task_ids_path,
        )
    if domain == "code_synthesis":
        if config.CODE_DATASET_NAME == "bigcodebench":
            return load_bigcodebench_tasks(
                num_tasks=num_tasks,
                seed=seed,
                start_index=start_index,
                task_ids=task_ids,
                task_ids_path=task_ids_path,
            )
        if config.CODE_DATASET_NAME == "humaneval":
            return load_humaneval_tasks(
                num_tasks=num_tasks,
                seed=seed,
                start_index=start_index,
                task_ids=task_ids,
                task_ids_path=task_ids_path,
            )
        if config.CODE_DATASET_NAME in {"custom_smoke", "code_smoke"}:
            from code_smoke_loader import load_code_smoke_tasks

            return load_code_smoke_tasks(
                num_tasks=num_tasks,
                seed=seed,
                start_index=start_index,
                task_ids=task_ids,
                task_ids_path=task_ids_path,
            )
        raise ValueError(f"Unsupported CODE_DATASET_NAME: {config.CODE_DATASET_NAME}")
    raise ValueError(f"Unsupported domain: {domain}")
