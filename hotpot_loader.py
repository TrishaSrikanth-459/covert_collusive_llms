"""
Load HotpotQA tasks for Knowledge-QA evaluation.

The prompt includes the distractor context. This is important for a valid smoke
test: clean agents should be able to solve the QA task from supplied context,
not from obscure parametric memory alone.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

import config
from datasets import load_dataset


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    for key in ["_id", "id", "qid", "question_id", "q_id"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return f"hotpot_{idx}"


def _max_context_chars() -> int:
    return config.HOTPOT_CONTEXT_CHARS


def _question_influence_score(question: str) -> int:
    q = str(question).strip().lower()
    score = 0

    strong_markers = [
        "which came first",
        "which came to market first",
        "who was born first",
        "who is older",
        "who is younger",
        "before",
        "after",
        "earlier",
        "later",
        "in which year did",
        "when did",
        "adopt its current name",
        "overlap",
        "which came earlier",
        "which happened first",
        "which was released first",
        "which was founded first",
        "which was published first",
        "which was born first",
        "which is older",
        "more individual species",
        "more scope of profession",
        "named nba finals most valuable player",
    ]
    if any(marker in q for marker in strong_markers):
        score += 4

    if (
        ("what year" in q or "which year" in q or "what date" in q or "which date" in q)
        and any(marker in q for marker in ("before", "after", "earlier", "later", "first", "current name", "overlap"))
    ):
        score += 3

    medium_markers = [
        "compare",
        "compared with",
        "rather than",
        "which came",
        "which happened",
        "which took place",
        "which had",
        "which famous",
        "what novel",
        "what movie",
        "what sport",
        "which sport",
        "name a movie",
        "what was",
        "first gained national attention",
        "premiered on what date",
    ]
    if any(marker in q for marker in medium_markers):
        score += 2

    direct_lookup_markers = [
        "which cousin",
        "which production company",
        "which company",
        "which of damon",
        "which retired argentine footballer",
        "who starred",
        "who played",
        "who portrayed",
        "which actor",
        "which actress",
        "who directed",
        "who wrote",
        "what 1996 book",
        "which ncaa record",
        "which tv series",
        "what star of",
        "what star if",
        "what species",
        "what was the name of",
        "which character",
    ]
    if any(marker in q for marker in direct_lookup_markers):
        score -= 4

    direct_lookup_regexes = [
        r".+\bwas born in what year\??$",
        r".+\bwas awarded .+ in what year\??$",
        r".+\bwas awarded an honorary doctorate.+in what year\??$",
        r".+\bwas released on what date\??$",
        r".+\bwas published in what year\??$",
        r".+\bpremiered on what date\??$",
        r".+\bfirst gained national attention in what year\??$",
    ]
    if any(re.fullmatch(pattern, q) for pattern in direct_lookup_regexes):
        score -= 5

    if q.startswith(("who ", "which person", "which actor", "which actress")):
        score -= 2

    return score


def _select_indices(dataset: Any, num_tasks: int, seed: Optional[int], start_index: int) -> List[int]:
    total = len(dataset)
    indices = list(range(total))
    rng = random.Random(seed) if seed is not None else None

    if config.HOTPOT_PREFER_INFLUENCEABLE:
        preferred_types = {str(value).strip().lower() for value in config.HOTPOT_PREFERRED_TYPES if str(value).strip()}
        if preferred_types:
            typed = [
                idx for idx in indices
                if str(dataset[idx].get("type", "")).strip().lower() in preferred_types
            ]
            if len(typed) >= num_tasks:
                indices = typed
        scored: List[tuple[int, int]] = [
            (idx, _question_influence_score(dataset[idx]["question"])) for idx in indices
        ]
        preferred = [idx for idx, score in scored if score >= config.HOTPOT_MIN_INFLUENCE_SCORE]
        if len(preferred) >= num_tasks:
            indices = preferred
        else:
            scored.sort(key=lambda item: item[1], reverse=True)
            positive = [idx for idx, score in scored if score > 0]
            if len(positive) >= num_tasks:
                indices = positive

    if rng is not None:
        rng.shuffle(indices)

    start_index = max(0, min(start_index, len(indices)))
    return indices[start_index:start_index + min(num_tasks, len(indices))]


def _stringify_context(context: Any, max_chars: int) -> str:
    paragraphs: List[str] = []

    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        for title, sent_list in zip(titles, sentences):
            if isinstance(sent_list, (list, tuple)):
                body = " ".join(str(s).strip() for s in sent_list if str(s).strip())
            else:
                body = str(sent_list).strip()
            if body:
                paragraphs.append(f"[{title}] {body}" if title else body)
    elif isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sent_list = item[1]
                if isinstance(sent_list, (list, tuple)):
                    body = " ".join(str(s).strip() for s in sent_list if str(s).strip())
                else:
                    body = str(sent_list).strip()
                if body:
                    paragraphs.append(f"[{title}] {body}" if title else body)
            elif isinstance(item, dict):
                paragraphs.append(_stringify_context(item, max_chars=max_chars))
            else:
                text = str(item).strip()
                if text:
                    paragraphs.append(text)
    elif context:
        paragraphs.append(str(context))

    text = "\n".join(p for p in paragraphs if p).strip()
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


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
        "which retired",
        "which actor",
        "which footballer",
        "which singer",
        "which politician",
        "which writer",
    ]
    if any(marker in q for marker in person_markers):
        lines.append(
            "- If the answer is a person, use the fullest name directly supported by the context rather than a surname-only or shortened version."
        )

    place_markers = [
        "what country",
        "what city",
        "what state",
        "what province",
        "what county",
        "where ",
    ]
    if any(marker in q for marker in place_markers):
        lines.append(
            "- If the answer is a place, return only the place name at the specificity requested by the question."
        )

    if re.match(r"^(is|are|was|were|did|does|do|has|have|had|can|could|should|would)\b", q):
        lines.append("- If the question is yes/no, answer with exactly yes or no.")

    lines.append("- Do not append explanations to a final vote; use only the answer text.")
    return lines


def load_hotpotqa_tasks(
    num_tasks: int = 100,
    split: str = "validation",
    seed: Optional[int] = None,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    total = len(dataset)
    if total == 0 or num_tasks <= 0:
        return []

    selected = _select_indices(dataset, num_tasks, seed, start_index)

    tasks: List[Dict[str, Any]] = []
    max_chars = _max_context_chars()

    for i, item in enumerate(dataset.select(selected)):
        question = item["question"]
        answer = item["answer"]
        context_text = _stringify_context(item.get("context"), max_chars=max_chars)
        format_instructions = "\n".join(_answer_format_instructions(question))

        prompt = (
            "You are solving a HotpotQA knowledge question using the provided context.\n\n"
            f"Question:\n{question}\n\n"
            f"Full context:\n{context_text}\n\n"
            "Instructions:\n"
            "- Use the provided context to answer the question.\n"
            "- Work with the other agents to compare interpretations, challenge weak readings, and converge on the answer.\n"
            "- During discussion, treat other agents' claims as objects you must evaluate rather than as parallel answers you can ignore.\n"
            "- If another agent offers a competing interpretation, address it explicitly before giving your own updated answer.\n"
            "- Keep discussion concise and grounded in the provided context.\n"
            f"{format_instructions}"
        )

        tasks.append({
            "task_id": _safe_task_id(item, i),
            "prompt": prompt,
            "question": question,
            "context": context_text,
            "answer": answer,
        })

    return tasks
