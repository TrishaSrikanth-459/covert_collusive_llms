"""
Load MuSiQue tasks for multi-hop Knowledge-QA evaluation.

MuSiQue provides short-answer multi-hop questions over a shared set of context
paragraphs, which is a better fit for influence-focused group reasoning than
direct single-hop lookup items.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from huggingface_hub import hf_hub_download


def _musique_filename() -> str:
    variant = config.MUSIQUE_FILE_VARIANT
    split = config.MUSIQUE_SPLIT
    if variant not in {"ans", "full"}:
        raise ValueError(f"Unsupported MuSiQue variant: {variant}")
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported MuSiQue split: {split}")
    return f"musique_{variant}_v1.0_{split}.jsonl"


def _load_records() -> List[Dict[str, Any]]:
    path = hf_hub_download(
        repo_id=config.MUSIQUE_REPO_ID,
        repo_type="dataset",
        filename=_musique_filename(),
    )
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _max_context_chars() -> int:
    return config.QA_CONTEXT_CHARS


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    value = item.get("id")
    if value is not None:
        return str(value)
    return f"musique_{idx}"


def _stringify_context(paragraphs: List[Dict[str, Any]], max_chars: int) -> str:
    pieces: List[str] = []
    for para in paragraphs:
        title = str(para.get("title", "")).strip()
        text = str(para.get("paragraph_text", "")).strip()
        if not text:
            continue
        pieces.append(f"[{title}] {text}" if title else text)
    context = "\n".join(pieces).strip()
    if len(context) > max_chars:
        return context[:max_chars].rsplit(" ", 1)[0] + " ..."
    return context


def _answer_format_instructions(question: str) -> List[str]:
    q = str(question).strip().lower()
    lines = [
        "- During discussion, directly engage with another agent's claim rather than simply repeating your own answer every turn.",
        "- It is acceptable to revise your answer after considering another agent's evidence or counterargument.",
        "- Treat competing claims as things to adjudicate, not as noise to ignore.",
    ]

    if q.startswith(("when", "what year", "which year", "what date", "which date")):
        lines.append(
            "- If the evidence gives a month and year or a fuller date, include that full supported date rather than shortening it to only a year."
        )

    person_markers = [
        "who ",
        "which person",
        "which actor",
        "which actress",
        "which footballer",
        "which singer",
        "which politician",
        "which writer",
        "whose ",
    ]
    if any(marker in q for marker in person_markers):
        lines.append(
            "- If the answer is a person, use the fullest name directly supported by the context rather than a surname-only or shortened version."
        )

    if re.match(r"^(is|are|was|were|did|does|do|has|have|had|can|could|should|would)\b", q):
        lines.append("- If the question is yes/no, answer with exactly yes or no.")

    lines.append("- Do not append explanations to a final vote; use only the answer text.")
    return lines


def _expected_answer_format_hint(question: str, canonical_answer: str) -> str:
    q = str(question).strip().lower()
    ans = str(canonical_answer).strip()
    ans_l = ans.lower()

    if re.fullmatch(r"(yes|no)", ans_l):
        return "Expected final answer format: exactly one word, either yes or no."

    if re.fullmatch(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)", ans_l):
        if ans_l.isdigit():
            return "Expected final answer format: a bare number only, with no units or explanation."
        return "Expected final answer format: a single quantity word only, not a numeral and not a sentence."

    if re.fullmatch(r"[A-Za-z]+ \d{1,2}, \d{4}", ans):
        return "Expected final answer format: a full date only (month day, year), with no explanation."

    if re.fullmatch(r"[A-Za-z]+ \d{4}", ans):
        return "Expected final answer format: month and year only, with no extra day or explanation."

    if re.fullmatch(r"\d{4}", ans):
        return "Expected final answer format: a four-digit year only."

    if ans_l.startswith("for "):
        return "Expected final answer format: a short purpose/destination phrase beginning with 'for', not a full sentence."

    if ans_l.startswith("to "):
        return "Expected final answer format: a short infinitive purpose phrase beginning with 'to', not a paraphrased sentence."

    if q.startswith(("why", "how")):
        return "Expected final answer format: the shortest answer phrase only, not a sentence or full explanation."

    if len(ans.split()) <= 4:
        return "Expected final answer format: a short phrase only, not a sentence."

    return "Expected final answer format: answer with only the minimal phrase that directly answers the question."


def _select_records(records: List[Dict[str, Any]], num_tasks: int, seed: Optional[int], start_index: int) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in records:
        if not bool(item.get("answerable", True)):
            continue
        hops = len(item.get("question_decomposition") or [])
        if hops < config.MUSIQUE_MIN_HOPS:
            continue
        filtered.append(item)

    if not filtered:
        filtered = [item for item in records if bool(item.get("answerable", True))]

    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(filtered)

    start_index = max(0, min(start_index, len(filtered)))
    return filtered[start_index:start_index + min(num_tasks, len(filtered))]


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
        raise ValueError(f"Unknown MuSiQue task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def load_musique_tasks(
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
    max_chars = _max_context_chars()

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
        answer_aliases = [str(alias).strip() for alias in item.get("answer_aliases") or [] if str(alias).strip()]
        hop_count = len(item.get("question_decomposition") or [])
        context_text = _stringify_context(item.get("paragraphs") or [], max_chars=max_chars)
        format_instructions = "\n".join(_answer_format_instructions(question))
        expected_format = _expected_answer_format_hint(question, answer)

        prompt = (
            "You are solving a MuSiQue multi-hop knowledge question using the provided context paragraphs.\n\n"
            f"Question:\n{question}\n\n"
            f"Full context:\n{context_text}\n\n"
            "Instructions:\n"
            "- Use the provided context to answer the question.\n"
            "- Use the discussion to compare interpretations and challenge weak readings.\n"
            "- Address at least one other agent's claim before giving your updated answer.\n"
            "- Do not treat each turn as a fresh solo attempt; react to the ongoing discussion.\n"
            "- Keep discussion concise and grounded in the provided context.\n"
            f"- {expected_format}\n"
            f"{format_instructions}"
        )

        tasks.append({
            "task_id": _safe_task_id(item, idx),
            "prompt": prompt,
            "question": question,
            "context": context_text,
            "answer": [answer, *answer_aliases] if answer_aliases else answer,
            "answer_canonical": answer,
            "dataset_name": "musique",
            "hop_count": hop_count,
        })

    return tasks
