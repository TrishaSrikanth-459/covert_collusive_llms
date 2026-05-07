"""
Dataset dispatch for the coordination-collusion benchmark.

The public repository intentionally keeps only the benchmark used for the
paper's final run. Earlier exploratory loaders for MuSiQue, HumanEval,
BigCodeBench, MRT/SHADE wrappers, and other smoke-test branches were removed
to keep the codebase reproducible and reviewable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from coordination_collusion_loader import load_coordination_collusion_tasks


def load_qa_tasks(
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return load_coordination_collusion_tasks(
        num_tasks=num_tasks,
        seed=seed,
        start_index=start_index,
        task_ids=task_ids,
        task_ids_path=task_ids_path,
    )


def load_domain_tasks(
    domain: str,
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if domain != "knowledge_qa":
        raise ValueError(
            "This cleaned repository supports only TASK_DOMAIN=knowledge_qa "
            "with QA_DATASET_NAME=coordination_collusion."
        )
    return load_qa_tasks(
        num_tasks=num_tasks,
        seed=seed,
        start_index=start_index,
        task_ids=task_ids,
        task_ids_path=task_ids_path,
    )
