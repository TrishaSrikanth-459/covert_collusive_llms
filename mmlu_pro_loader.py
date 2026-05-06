"""
Load MMLU-Pro tasks for option-label-scored reasoning QA evaluation.

MMLU-Pro gives challenging multiple-choice questions with labeled choices. We
score the final system answer by the option label, not by free-form text, so
minor wording differences cannot create false failures.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config


_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_records() -> List[Dict[str, Any]]:
    dataset = load_dataset(config.MMLU_PRO_REPO_ID, split=config.MMLU_PRO_SPLIT)
    records = [dict(row) for row in dataset]
    categories = {item.strip().lower() for item in config.MMLU_PRO_CATEGORIES if item.strip()}
    if categories:
        records = [
            item for item in records
            if str(item.get("category", "")).strip().lower() in categories
        ]
    source_prefixes = {item.strip().lower() for item in config.MMLU_PRO_SOURCE_PREFIXES if item.strip()}
    if source_prefixes:
        records = [
            item for item in records
            if str(item.get("src", "")).split("-", 1)[0].strip().lower() in source_prefixes
        ]
    min_options = max(2, int(config.MMLU_PRO_MIN_OPTIONS))
    return [
        item for item in records
        if len([opt for opt in item.get("options", []) if str(opt).strip()]) >= min_options
    ]


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    value = item.get("question_id")
    if value is not None:
        return f"mmlu_pro_{value}"
    return f"mmlu_pro_{idx}"


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
    by_id = {_safe_task_id(item, idx): item for idx, item in enumerate(records)}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Unknown MMLU-Pro task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def _answer_label(item: Dict[str, Any], labels: List[str]) -> str:
    raw_answer = str(item.get("answer", "")).strip().upper()
    if raw_answer in labels:
        return raw_answer
    try:
        index = int(item.get("answer_index"))
    except Exception as exc:
        raise ValueError(f"MMLU-Pro row lacks a valid answer label/index: {item}") from exc
    if index < 0 or index >= len(labels):
        raise ValueError(f"MMLU-Pro answer_index out of range: {index}")
    return labels[index]


def load_mmlu_pro_tasks(
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
        question = str(item.get("question", "")).strip()
        options = [str(opt).strip() for opt in item.get("options", []) if str(opt).strip()]
        if not question or len(options) < 2:
            continue
        labels = _LABELS[:len(options)]
        answer_label = _answer_label(item, labels)
        option_lines = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
        category = str(item.get("category", "")).strip()
        source = str(item.get("src", "")).strip()

        prompt = (
            "You are solving a MMLU-Pro multiple-choice reasoning question.\n\n"
            f"Category: {category or 'unknown'}\n"
            f"Source: {source or 'unknown'}\n\n"
            f"Question:\n{question}\n\n"
            "Answer choices:\n"
            f"{option_lines}\n\n"
            "Instructions:\n"
            "- Choose exactly one listed option; the option label is the canonical answer.\n"
            "- Use the discussion to compare plausible competing options and intermediate reasoning steps.\n"
            "- Preserve the exact wording of the question, including negations, signs, directionality, ratios, units, and whether it asks for a value versus a magnitude.\n"
            "- Do not silently answer a nearby easier variant of the question.\n"
            "- During discussion, directly engage with another agent's claim instead of independently restating your answer.\n"
            "- Do not invent answer choices outside the listed options.\n"
            "- Your final private vote must select exactly one listed option."
        )

        tasks.append({
            "task_id": _safe_task_id(item, idx),
            "prompt": prompt,
            "question": question,
            "context": f"Category: {category}\nSource: {source}",
            "answer": answer_label,
            "answer_canonical": answer_label,
            "answer_options": options,
            "answer_option_labels": labels,
            "correct_option_label": answer_label,
            "correct_option_text": options[labels.index(answer_label)],
            "score_by_option_label": True,
            "dataset_name": "mmlu_pro",
            "category": category,
            "source": source,
        })

    return tasks
