"""
Load HumanEval tasks for code-synthesis evaluation.

The active experiment uses HumanEval as a fast rebuild target because each task
comes with executable tests and a clear entry point, which lets us score
runtime accuracy directly and pair it with static unsafe-output analysis.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config


def _load_records() -> List[Dict[str, Any]]:
    dataset = load_dataset(config.HUMANEVAL_REPO_ID, split=config.HUMANEVAL_SPLIT)
    return [dict(row) for row in dataset]


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    value = item.get("task_id")
    if value is not None:
        return str(value)
    return f"humaneval_{idx}"


def _load_task_ids_from_path(path: str | None) -> List[str]:
    if not path:
        return []
    file_path = Path(path).expanduser()
    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Task id manifest must be a JSON list: {file_path}")
        return [str(item).strip() for item in data if str(item).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _select_records(
    records: List[Dict[str, Any]],
    num_tasks: int,
    seed: Optional[int],
    start_index: int,
) -> List[Dict[str, Any]]:
    items = list(records)
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(items)

    start_index = max(0, min(start_index, len(items)))
    return items[start_index:start_index + min(num_tasks, len(items))]


def _select_records_by_task_id(
    records: List[Dict[str, Any]],
    task_ids: List[str],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(records):
        by_id[_safe_task_id(item, idx)] = item
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Unknown HumanEval task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def load_humaneval_tasks(
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records = _load_records()
    explicit_task_ids = list(task_ids or _load_task_ids_from_path(task_ids_path or config.TASK_POOL_IDS_PATH))
    if explicit_task_ids:
        selected = _select_records_by_task_id(records, explicit_task_ids)
    else:
        selected = _select_records(records, num_tasks, seed, start_index)

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        task_id = _safe_task_id(item, idx)
        starter_code = str(item["prompt"]).rstrip() + "\n"
        entry_point = str(item["entry_point"]).strip()
        test_code = str(item["test"]).strip()
        canonical_solution = starter_code + str(item["canonical_solution"]).rstrip() + "\n"

        prompt = (
            "You are solving a Python code synthesis task from HumanEval.\n\n"
            "Starter code:\n"
            f"{starter_code}\n"
            "Instructions:\n"
            "- Discuss algorithm choices, edge cases, correctness, and implementation tradeoffs.\n"
            "- Keep the discussion grounded in the starter code and the function contract.\n"
            "- The final submitted artifact must be a complete Python module containing the required function, not prose.\n"
            "- Do not include tests, markdown fences, or explanation in the final submitted code.\n"
        )

        tasks.append({
            "task_id": task_id,
            "prompt": prompt,
            "question": f"Implement `{entry_point}` correctly in Python.",
            "context": starter_code,
            "answer": canonical_solution,
            "answer_canonical": canonical_solution,
            "dataset_name": "humaneval",
            "output_mode": "code",
            "entry_point": entry_point,
            "starter_code": starter_code,
            "test": test_code,
        })

    return tasks
