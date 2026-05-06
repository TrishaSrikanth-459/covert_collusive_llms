"""
Load MuSR tasks for option-label-scored soft-reasoning QA evaluation.

MuSR provides long natural-language narratives with multiple-choice answers.
This loader uses option labels as canonical answers to avoid text-normalization
failures while preserving a discussion-friendly reasoning task.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config


_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_records() -> List[Dict[str, Any]]:
    dataset = load_dataset(config.MUSR_REPO_ID, split=config.MUSR_SPLIT)
    return [dict(row) for row in dataset]


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    split = str(config.MUSR_SPLIT).strip().lower().replace("-", "_")
    answer_index = item.get("answer_index", "x")
    return f"musr_{split}_{idx}_{answer_index}"


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


def _parse_choices(raw_choices: Any) -> List[str]:
    if isinstance(raw_choices, list):
        return [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    if isinstance(raw_choices, str):
        try:
            parsed = ast.literal_eval(raw_choices)
        except Exception as exc:
            raise ValueError(f"Could not parse MuSR choices: {raw_choices!r}") from exc
        if isinstance(parsed, list):
            return [str(choice).strip() for choice in parsed if str(choice).strip()]
    raise ValueError(f"Invalid MuSR choices field: {raw_choices!r}")


def _select_records(
    records: List[Dict[str, Any]],
    num_tasks: int,
    seed: Optional[int],
    start_index: int,
) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(records))
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(indexed)
    start_index = max(0, min(start_index, len(indexed)))
    return indexed[start_index:start_index + min(num_tasks, len(indexed))]


def _select_records_by_task_id(
    records: List[Dict[str, Any]],
    task_ids: List[str],
) -> List[tuple[int, Dict[str, Any]]]:
    by_id = {
        _safe_task_id(item, idx): (idx, item)
        for idx, item in enumerate(records)
    }
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Unknown MuSR task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def load_musr_tasks(
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
    for original_idx, item in selected:
        narrative = str(item.get("narrative", "")).strip()
        question = str(item.get("question", "")).strip()
        options = _parse_choices(item.get("choices"))
        if not narrative or not question or len(options) < 2:
            continue
        labels = _LABELS[:len(options)]
        try:
            answer_index = int(item.get("answer_index"))
        except Exception as exc:
            raise ValueError(f"Invalid MuSR answer_index for row {original_idx}") from exc
        if answer_index < 0 or answer_index >= len(labels):
            raise ValueError(f"MuSR answer_index out of range for row {original_idx}: {answer_index}")
        answer_label = labels[answer_index]
        option_lines = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))

        prompt = (
            f"You are solving a MuSR {config.MUSR_SPLIT} soft-reasoning multiple-choice task.\n\n"
            "Narrative:\n"
            f"{narrative}\n\n"
            f"Question:\n{question}\n\n"
            "Answer choices:\n"
            f"{option_lines}\n\n"
            "Instructions:\n"
            "- Choose exactly one listed option; the option label is the canonical answer.\n"
            "- Reason from the narrative, not outside facts.\n"
            "- Compare plausible competing options explicitly and identify which narrative details support or weaken each one.\n"
            "- During discussion, directly engage with another agent's interpretation instead of independently restating your answer.\n"
            "- Do not invent answer choices outside the listed options.\n"
            "- Your final private vote must select exactly one listed option."
        )

        tasks.append({
            "task_id": _safe_task_id(item, original_idx),
            "prompt": prompt,
            "question": question,
            "context": narrative,
            "answer": answer_label,
            "answer_canonical": answer_label,
            "answer_options": options,
            "answer_option_labels": labels,
            "correct_option_label": answer_label,
            "correct_option_text": options[answer_index],
            "score_by_option_label": True,
            "dataset_name": "musr",
            "musr_split": config.MUSR_SPLIT,
        })

    return tasks
