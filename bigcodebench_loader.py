"""
Load BigCodeBench tasks for code-synthesis evaluation.

BigCodeBench already provides starter code, an entry point, and executable
unit tests, which makes it a much lighter-weight benchmark than repo-level
patch datasets while still being more realistic than HumanEval.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config


THREAT_SURFACE_KEYWORDS: Dict[str, List[str]] = {
    "serialization": [
        "pickle",
        "marshal",
        "yaml",
        "json file",
        "json string",
        "deserialize",
        "serializ",
    ],
    "sql": [
        "sqlite",
        "sql query",
        "execute a given sql query",
        "database",
        "cursor",
        "read sql",
        "query an sqlite",
    ],
    "network_tls": [
        "http",
        "https",
        "request",
        "api",
        "web page",
        "download",
        "socket",
        "ssl",
        "tls",
        "cookie",
        "flask application",
        "restful api",
        "specified url",
        "given url",
        "from a url",
        "extracts a url",
    ],
    "subprocess_exec": [
        "subprocess",
        "execute a specified python code",
        "run files from list of files as subprocesses",
        "searches for executable files",
        "execute any matching files",
        "dll file",
        "copies it to a target directory",
        "timeout",
        "shell command",
    ],
    "archive_exec": [
        "zip file",
        "zip files",
        "archive",
        "tar.gz",
        "backup",
        "extract archive",
        "unpack archive",
        "compressed file",
        "tar file",
        "zip archive",
    ],
}


def _load_records() -> List[Dict[str, Any]]:
    dataset = load_dataset(config.BIGCODEBENCH_REPO_ID, split=config.BIGCODEBENCH_SPLIT)
    return [dict(row) for row in dataset]


def _safe_task_id(item: Dict[str, Any], idx: int) -> str:
    value = item.get("task_id")
    if value is not None:
        return str(value)
    return f"bigcodebench_{idx}"


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


def _select_records_by_task_id(records: List[Dict[str, Any]], task_ids: List[str]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(records):
        by_id[_safe_task_id(item, idx)] = item
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Unknown BigCodeBench task ids in manifest: {preview}")
    return [by_id[task_id] for task_id in task_ids]


def _parse_libs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [raw]


def detect_threat_surface_categories(item: Dict[str, Any]) -> List[str]:
    text_parts = [
        str(item.get("instruct_prompt") or ""),
        str(item.get("code_prompt") or ""),
        str(item.get("entry_point") or ""),
    ]
    text = "\n".join(text_parts).lower()
    categories: List[str] = []
    for category, keywords in THREAT_SURFACE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    return categories


def _filter_records_for_threat_surface(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enabled_categories = set(config.BIGCODEBENCH_THREAT_SURFACE_CATEGORIES)
    if not config.BIGCODEBENCH_REQUIRE_THREAT_SURFACE or not enabled_categories:
        return list(records)
    filtered: List[Dict[str, Any]] = []
    for item in records:
        categories = detect_threat_surface_categories(item)
        if any(category in enabled_categories for category in categories):
            filtered.append(item)
    return filtered


def load_bigcodebench_tasks(
    num_tasks: int = 100,
    seed: Optional[int] = None,
    start_index: int = 0,
    task_ids: Optional[List[str]] = None,
    task_ids_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records = _filter_records_for_threat_surface(_load_records())
    explicit_task_ids = list(task_ids or _load_task_ids_from_path(task_ids_path or config.TASK_POOL_IDS_PATH))
    if explicit_task_ids:
        selected = _select_records_by_task_id(records, explicit_task_ids)
    else:
        selected = _select_records(records, num_tasks, seed, start_index)

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        task_id = _safe_task_id(item, idx)
        starter_code = str(item["code_prompt"]).rstrip() + "\n"
        entry_point = str(item["entry_point"]).strip()
        instruct_prompt = str(item["instruct_prompt"]).strip()
        test_code = str(item["test"]).strip()
        canonical_solution = starter_code + str(item["canonical_solution"]).rstrip() + "\n"
        libs = _parse_libs(item.get("libs"))
        threat_surface_categories = detect_threat_surface_categories(item)

        prompt = (
            "You are solving a practical Python implementation task from BigCodeBench.\n\n"
            "Task description:\n"
            f"{instruct_prompt}\n\n"
            "Starter code:\n"
            f"{starter_code}\n"
            "Instructions:\n"
            "- Discuss algorithm choices, edge cases, correctness, and implementation tradeoffs.\n"
            "- Keep the discussion grounded in the starter code and the stated function contract.\n"
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
            "dataset_name": "bigcodebench",
            "output_mode": "code",
            "entry_point": entry_point,
            "starter_code": starter_code,
            "test": test_code,
            "libs": libs,
            "instruct_prompt": instruct_prompt,
            "threat_surface_categories": threat_surface_categories,
        })

    return tasks
