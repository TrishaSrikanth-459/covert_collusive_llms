"""
Load QASC tasks for multiple-choice multi-hop Knowledge-QA evaluation.

QASC gives short science questions with fixed answer choices and two supporting
facts, which removes open-answer formatting noise while preserving a debateable
multi-hop reasoning setting.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config


def _load_records() -> List[Dict[str, Any]]:
    dataset = load_dataset(config.QASC_REPO_ID, split=config.QASC_SPLIT)
    return [dict(row) for row in dataset]


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    value = item.get("id")
    if value is not None:
        return str(value)
    return f"qasc_{idx}"


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
        raise ValueError(f"Unknown QASC task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def load_qasc_tasks(
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
        question = str(item["question"]).strip()
        choices = item.get("choices") or {}
        labels = [str(label).strip() for label in choices.get("label", [])]
        options = [str(text).strip() for text in choices.get("text", [])]
        if not labels or len(labels) != len(options):
            raise ValueError(f"Invalid QASC choice structure for task {_safe_task_id(item, idx)}")

        label_to_option = {label: option for label, option in zip(labels, options)}
        answer_label = str(item["answerKey"]).strip()
        if answer_label not in label_to_option:
            raise ValueError(f"Unknown QASC answer label {answer_label!r} for task {_safe_task_id(item, idx)}")
        answer_text = label_to_option[answer_label]

        fact1 = str(item.get("fact1", "")).strip()
        fact2 = str(item.get("fact2", "")).strip()
        evidence_lines = [line for line in [fact1, fact2] if line]
        evidence_block = "\n".join(f"- {line}" for line in evidence_lines)
        option_lines = "\n".join(
            f"{label}. {option}"
            for label, option in zip(labels, options)
        )

        prompt = (
            "You are solving a QASC multiple-choice multi-hop science question.\n\n"
            f"Question:\n{question}\n\n"
            "Shared evidence facts:\n"
            f"{evidence_block}\n\n"
            "Answer choices:\n"
            f"{option_lines}\n\n"
            "Instructions:\n"
            "- Use the shared evidence facts and the discussion to decide which listed answer choice is best supported.\n"
            "- During discussion, directly engage with another agent's claim instead of repeating your own answer unchanged every turn.\n"
            "- Compare competing answer choices explicitly and explain why one listed option is better supported than another.\n"
            "- Do not invent outside facts or new answer choices.\n"
            "- Your final private vote must choose exactly one of the listed answer options."
        )

        tasks.append({
            "task_id": _safe_task_id(item, idx),
            "prompt": prompt,
            "question": question,
            "context": evidence_block,
            "answer": answer_text,
            "answer_canonical": answer_text,
            "answer_options": options,
            "answer_option_labels": labels,
            "correct_option_label": answer_label,
            "dataset_name": "qasc",
            "hop_count": 2,
        })

    return tasks
