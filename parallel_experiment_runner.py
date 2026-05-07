"""
Main runner for the collusive covert-attack experiment.

Design implemented here:
- fixed attack strategy: fake injection
- default scale: 25 tasks x 5 repetitions x 9 conditions = 1125 task runs
- resumability: each completed task is checkpointed to JSONL before aggregate
  rows are written, so interrupted runs continue at the next unfinished task.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as cfg
from agent import WorkerAgent
from detector_agent import DetectorAgent
from evaluation import Evaluator
from models import AttackType
from prompt_injection import inject_hidden_prompts
from qa_loader import load_domain_tasks
from role_assigner import assign_roles, malicious_count_from_label, mark_malicious
from simulation import Simulation


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "run"


def _protocol_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    active_files = [
        "agent.py",
        "detector_agent.py",
        "simulation.py",
        "evaluation.py",
        "permission_manager.py",
        "prompt_injection.py",
        "coordination_collusion_loader.py",
        "qa_loader.py",
        "role_assigner.py",
        "config.py",
        "parallel_experiment_runner.py",
    ]
    digest = hashlib.sha1()
    for rel_path in active_files:
        path = root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


DOMAIN = cfg.TASK_DOMAIN
FIXED_ATTACK_TYPE = AttackType.FAKE_INJECTION
CONDITION_SPECS = [
    ("clean_baseline", "0", 0),
    ("single_attacker", "1", 0),
    ("collusive_attack", "2", 0),
    ("clean_one_detector", "0", 1),
    ("single_attacker_one_detector", "1", 1),
    ("collusive_attack_one_detector", "2", 1),
    ("clean_two_detectors", "0", 2),
    ("single_attacker_two_detectors", "1", 2),
    ("collusive_attack_two_detectors", "2", 2),
]

MAX_CONCURRENT = _get_env_int("MAX_CONCURRENT", 4)
MAX_CONCURRENT_CAP = _get_env_int("MAX_CONCURRENT_CAP", 12)
TOTAL_REPS = cfg.TOTAL_REPS
RUN_LABEL = _slugify(os.getenv("RUN_LABEL", f"coordination_collusion_{DOMAIN}_full9_rep5_w5"))
EXPERIMENT_VERSION = os.getenv("EXPERIMENT_VERSION", f"2026-05-coordination-collusion-{DOMAIN}-full9-w5-v1")
PROTOCOL_FINGERPRINT = os.getenv("PROTOCOL_FINGERPRINT", _protocol_fingerprint())
OUTPUT_CSV = os.getenv("OUTPUT_CSV", f"experiment_results.{RUN_LABEL}.csv")
TASK_PROGRESS_JSONL = os.getenv("TASK_PROGRESS_JSONL", f"task_progress.{RUN_LABEL}.jsonl")
FAILED_CONFIGS_CSV = os.getenv("FAILED_CONFIGS_CSV", f"failed_configs.{RUN_LABEL}.csv")
CONFIG_RETRY_LIMIT = _get_env_int("CONFIG_RETRY_LIMIT", 5)
CONFIG_RETRY_BASE_DELAY = _get_env_float("CONFIG_RETRY_BASE_DELAY", 5.0)
CONFIG_RETRY_MAX_DELAY = _get_env_float("CONFIG_RETRY_MAX_DELAY", 180.0)

AGGREGATE_METRICS = [
    "unsafe_output_rate",
    "functional_correctness",
    "completion_failure",
    "attack_success",
    "false_positive_rate",
    "detection_rate",
    "detection_precision",
    "attribution_accuracy",
    "detection_latency",
    "benign_restriction_rate",
    "benign_recovery_time",
    "wall_time_seconds",
]

DETECTION_METRICS = {
    "false_positive_rate",
    "detection_rate",
    "detection_precision",
    "attribution_accuracy",
    "detection_latency",
    "benign_restriction_rate",
    "benign_recovery_time",
}

BASE_OUTPUT_FIELDNAMES = [
    "run_label",
    "experiment_version",
    "protocol_fingerprint",
    "global_seed",
    "rep",
    "condition_id",
    "condition_name",
    "attack_type",
    "m",
    "malicious_count",
    "d",
    "domain",
    "num_workers",
    "task_domain",
    "qa_dataset",
    "qa_tasks",
    "qa_context_chars",
    "total_turns",
    "temperature",
    "max_tokens",
    "model",
    "credibility_full",
    "credibility_restricted",
    "credibility_quarantine",
    "detector_action_min_confidence",
    "clean_turns_for_recovery",
    "recovery_increment",
    "is_clean_baseline",
    "is_single_attacker",
    "is_collusive",
    "is_all_malicious",
    "has_detectors",
]
OUTPUT_FIELDNAMES = BASE_OUTPUT_FIELDNAMES[:]
for metric_name in AGGREGATE_METRICS:
    OUTPUT_FIELDNAMES.extend([f"{metric_name}_mean", f"{metric_name}_std"])
OUTPUT_FIELDNAMES.extend(["detection_tasks_evaluated", "detection_tasks_excluded"])
OUTPUT_FIELDNAMES.extend(["detection_latency_tasks_evaluated", "detection_latency_tasks_excluded"])
OUTPUT_FIELDNAMES.append("task_pool_signature")
OUTPUT_FIELDNAMES.append("task_pool_ids_path")
OUTPUT_FIELDNAMES.append("tasks_evaluated")


@dataclass(frozen=True)
class ExperimentConfig:
    rep: int
    m: str
    d: int
    attack_type: AttackType = FIXED_ATTACK_TYPE
    domain: str = DOMAIN
    condition_name: str = ""

    @property
    def condition_id(self) -> str:
        return f"{self.condition_name}__m{self.m}_d{self.d}"
def _runtime_context() -> Dict[str, Any]:
    return {
        "run_label": RUN_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "global_seed": cfg.GLOBAL_SEED,
        "num_workers": cfg.NUM_WORKERS,
        "task_domain": cfg.TASK_DOMAIN,
        "qa_dataset": cfg.QA_DATASET_NAME,
        "qa_tasks": cfg.QA_TASKS,
        "qa_context_chars": cfg.QA_CONTEXT_CHARS,
        "task_pool_ids_path": cfg.TASK_POOL_IDS_PATH,
        "total_turns": cfg.TOTAL_TURNS,
        "temperature": cfg.TEMPERATURE,
        "max_tokens": cfg.MAX_TOKENS,
        "model": cfg.MODEL_NAME,
        "credibility_full": cfg.CREDIBILITY_FULL,
        "credibility_restricted": cfg.CREDIBILITY_RESTRICTED,
        "credibility_quarantine": cfg.CREDIBILITY_QUARANTINE,
        "detector_action_min_confidence": cfg.DETECTOR_ACTION_MIN_CONFIDENCE,
        "clean_turns_for_recovery": cfg.CLEAN_TURNS_FOR_RECOVERY,
        "recovery_increment": cfg.RECOVERY_INCREMENT,
    }


def _condition_labels(exp_config: ExperimentConfig, malicious_count: Optional[int] = None) -> Dict[str, Any]:
    if malicious_count is None:
        malicious_count = cfg.NUM_WORKERS if exp_config.m == "all" else int(exp_config.m)
    return {
        "is_clean_baseline": malicious_count == 0 and exp_config.d == 0,
        "is_single_attacker": malicious_count == 1,
        "is_collusive": malicious_count >= 2 and exp_config.m != "all",
        "is_all_malicious": exp_config.m == "all",
        "has_detectors": exp_config.d > 0,
    }


def _config_key(exp_config: ExperimentConfig, task_pool_signature: str = "") -> str:
    payload = {
        "rep": exp_config.rep,
        "attack_type": exp_config.attack_type.value,
        "m": exp_config.m,
        "d": exp_config.d,
        "domain": exp_config.domain,
        "condition_name": exp_config.condition_name,
        "task_pool_signature": task_pool_signature,
        **_runtime_context(),
    }
    return json.dumps(payload, sort_keys=True)


def _config_key_from_row(row: Dict[str, Any]) -> str:
    payload = {
        "rep": int(row["rep"]),
        "attack_type": row["attack_type"],
        "m": row["m"],
        "d": int(row["d"]),
        "domain": row["domain"],
        "condition_name": row["condition_name"],
        "task_pool_signature": row.get("task_pool_signature", ""),
        "run_label": row.get("run_label", RUN_LABEL),
        "experiment_version": row.get("experiment_version", EXPERIMENT_VERSION),
        "protocol_fingerprint": row.get("protocol_fingerprint", PROTOCOL_FINGERPRINT),
        "global_seed": int(float(row.get("global_seed", cfg.GLOBAL_SEED))),
        "num_workers": int(float(row.get("num_workers", cfg.NUM_WORKERS))),
        "task_domain": row.get("task_domain", cfg.TASK_DOMAIN),
        "qa_dataset": row.get("qa_dataset", cfg.QA_DATASET_NAME),
        "qa_tasks": int(float(row.get("qa_tasks", cfg.QA_TASKS))),
        "qa_context_chars": int(float(row.get("qa_context_chars", cfg.QA_CONTEXT_CHARS))),
        "task_pool_ids_path": row.get("task_pool_ids_path", cfg.TASK_POOL_IDS_PATH),
        "total_turns": int(float(row.get("total_turns", cfg.TOTAL_TURNS))),
        "temperature": float(row.get("temperature", cfg.TEMPERATURE)),
        "max_tokens": int(float(row.get("max_tokens", cfg.MAX_TOKENS))),
        "model": row.get("model", cfg.MODEL_NAME),
        "credibility_full": float(row.get("credibility_full", cfg.CREDIBILITY_FULL)),
        "credibility_restricted": float(row.get("credibility_restricted", cfg.CREDIBILITY_RESTRICTED)),
        "credibility_quarantine": float(row.get("credibility_quarantine", cfg.CREDIBILITY_QUARANTINE)),
        "detector_action_min_confidence": float(row.get("detector_action_min_confidence", cfg.DETECTOR_ACTION_MIN_CONFIDENCE)),
        "clean_turns_for_recovery": int(float(row.get("clean_turns_for_recovery", cfg.CLEAN_TURNS_FOR_RECOVERY))),
        "recovery_increment": float(row.get("recovery_increment", cfg.RECOVERY_INCREMENT)),
    }
    return json.dumps(payload, sort_keys=True)


def _task_signature(task: Dict[str, Any]) -> str:
    payload = {
        "task_id": task.get("task_id"),
        "question": task.get("question", ""),
        "answer": task.get("answer", ""),
        "answer_options": task.get("answer_options", []),
        "answer_option_labels": task.get("answer_option_labels", []),
        "attack_target_answer": task.get("attack_target_answer", ""),
        "attack_target_option": task.get("attack_target_option", ""),
        "backdoor_source": task.get("backdoor_source", ""),
        "backdoor_model": task.get("backdoor_model", ""),
        "starter_code": task.get("starter_code", ""),
        "test": task.get("test", ""),
        "output_mode": task.get("output_mode", ""),
        "prompt": task.get("prompt", ""),
        "context": task.get("context", ""),
        "agent_private_contexts": task.get("agent_private_contexts", {}),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _task_pool_signature(tasks: List[Dict[str, Any]]) -> str:
    payload = [(_task_signature(task), task.get("task_id")) for task in tasks]
    raw = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _task_key(exp_config: ExperimentConfig, task: Dict[str, Any], task_pool_signature: str = "") -> str:
    return f"{_config_key(exp_config, task_pool_signature)}::{task.get('task_id')}::{_task_signature(task)}"


def generate_configs(total_reps: int = None) -> List[ExperimentConfig]:
    reps = total_reps or TOTAL_REPS
    configs: List[ExperimentConfig] = []
    for condition_name, m_label, detectors in CONDITION_SPECS:
        for rep in range(1, reps + 1):
            configs.append(
                ExperimentConfig(
                    rep=rep,
                    m=m_label,
                    d=detectors,
                    condition_name=condition_name,
                )
            )
    return configs


def generate_smoke_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(1, "0", 0, condition_name="clean_baseline"),
        ExperimentConfig(1, "1", 0, condition_name="single_attacker"),
        ExperimentConfig(1, "1", 1, condition_name="single_attacker_one_detector"),
        ExperimentConfig(1, "2", 0, condition_name="collusive_attack"),
        ExperimentConfig(1, "2", 2, condition_name="collusive_attack_two_detectors"),
    ]


def _row_for_output(
    exp_config: ExperimentConfig,
    result: Dict[str, Any],
    malicious_count: int,
    task_pool_signature: str,
) -> Dict[str, Any]:
    row = asdict(exp_config)
    row["attack_type"] = exp_config.attack_type.value
    row["condition_id"] = exp_config.condition_id
    row["malicious_count"] = malicious_count
    row.update(_runtime_context())
    row["task_pool_signature"] = task_pool_signature
    row.update(_condition_labels(exp_config, malicious_count))
    row.update(result)
    return row


def _load_completed_configs(csv_path: str) -> set[str]:
    completed: set[str] = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(_config_key_from_row(row))
            except Exception:
                continue
    return completed


def _validate_output_csv_header(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header != OUTPUT_FIELDNAMES:
        raise ValueError(
            "Existing output CSV header does not match the current experiment schema. "
            "Use a fresh --run-label/--output-csv or remove the stale CSV before running."
        )


def _load_task_progress(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    progress: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        return progress
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "task_key" in rec and "result" in rec:
                    progress[rec["task_key"]] = rec["result"]
            except Exception:
                continue
    return progress


async def _append_task_progress(jsonl_path: str, lock: asyncio.Lock, task_key: str, result: Dict[str, Any]) -> None:
    async with lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_key": task_key, "result": result}, ensure_ascii=False) + "\n")
            f.flush()


def _build_task_pools(seed: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    return {
        DOMAIN: load_domain_tasks(
            DOMAIN,
            cfg.QA_TASKS,
            seed=seed,
            task_ids_path=cfg.TASK_POOL_IDS_PATH or None,
        )
    }


def _collect_numeric(results: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for result in results:
        if key not in result:
            continue
        try:
            value = float(result[key])
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        values.append(value)
    return values


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    detection_scoped_results = [result for result in results if result.get("_detection_stats_valid", True)]
    out["detection_tasks_evaluated"] = len(detection_scoped_results)
    out["detection_tasks_excluded"] = len(results) - len(detection_scoped_results)
    latency_scoped_results = [result for result in detection_scoped_results if math.isfinite(float(result.get("detection_latency", float("nan"))))]
    out["detection_latency_tasks_evaluated"] = len(latency_scoped_results)
    out["detection_latency_tasks_excluded"] = len(results) - len(latency_scoped_results)
    for key in AGGREGATE_METRICS:
        scoped_results = results
        if key in DETECTION_METRICS:
            scoped_results = detection_scoped_results
            if not scoped_results:
                out[f"{key}_mean"] = float("nan")
                out[f"{key}_std"] = 0.0
                continue
        vals = _collect_numeric(scoped_results, key)
        if key == "detection_latency":
            vals = _collect_numeric(latency_scoped_results, key)
            if not vals:
                out[f"{key}_mean"] = float("nan")
                out[f"{key}_std"] = 0.0
                continue
        if not vals:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = 0.0
            continue
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    out["tasks_evaluated"] = len(results)
    return out


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = [
        "429",
        "rate limit",
        "ratelimit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    ]
    return any(marker in text for marker in retry_markers)


def _backoff_seconds(attempt: int) -> float:
    delay = min(CONFIG_RETRY_MAX_DELAY, CONFIG_RETRY_BASE_DELAY * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(3.0, delay * 0.25))
    return delay + jitter


def _build_task_rng(exp_config: ExperimentConfig, task: Dict[str, Any], task_index: int) -> random.Random:
    seed_material = "::".join([
        str(cfg.GLOBAL_SEED),
        str(exp_config.rep),
        exp_config.domain,
        str(task.get("task_id", task_index)),
        exp_config.attack_type.value,
        exp_config.m,
        str(exp_config.d),
        exp_config.condition_name,
    ])
    return random.Random(seed_material)


def _stable_experiment_id(exp_config: ExperimentConfig, task: Dict[str, Any], task_index: int) -> str:
    task_sig = _task_signature(task)
    raw = f"{RUN_LABEL}_{exp_config.condition_id}_rep{exp_config.rep}_{task_index}_{task.get('task_id', task_index)}_{task_sig}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    slug = _slugify(raw)
    max_slug_len = 180 - len(digest) - 1
    return f"exp_{slug[:max_slug_len]}_{digest}"


async def _run_single_task(exp_config: ExperimentConfig, task: Dict[str, Any], task_index: int) -> Dict[str, Any]:
    started = time.time()
    task_rng = _build_task_rng(exp_config, task, task_index)
    experiment_id = _stable_experiment_id(exp_config, task, task_index)
    task_desc = task["prompt"]

    agents_data = assign_roles(exp_config.domain, rng=task_rng)
    malicious_count = malicious_count_from_label(exp_config.m, len(agents_data))
    agents_data = mark_malicious(
        agents_data,
        exp_config.m,
        rng=task_rng,
    )

    malicious_ids = [agent.id for agent in agents_data if agent.is_malicious]
    all_agent_ids = [agent.id for agent in agents_data]
    detector_ids = [1000 + i for i in range(exp_config.d)] if exp_config.d > 0 else []

    if malicious_count > 0:
        agents_data = inject_hidden_prompts(
            agents_data,
            exp_config.m,
            exp_config.attack_type,
            domain=exp_config.domain,
            task_metadata=task,
        )

    workers = [
        WorkerAgent(
            agent_data,
            task_desc,
            task_metadata=task,
        )
        for agent_data in agents_data
    ]

    detectors: List[DetectorAgent] = []
    if exp_config.d > 0:
        db_path = os.path.join(cfg.LOG_DIR, f"{experiment_id}.db")
        for detector_id in detector_ids:
            detectors.append(
                DetectorAgent(
                    detector_id,
                    db_path,
                    exp_config.domain,
                    None,
                    None,
                    detector_ids=detector_ids,
                    task_metadata=task,
                )
            )

    metadata = {
        "experiment_id": experiment_id,
        "condition_id": exp_config.condition_id,
        "condition_name": exp_config.condition_name,
        "domain": exp_config.domain,
        "m": exp_config.m,
        "malicious_count": malicious_count,
        "d": exp_config.d,
        "attack_type": exp_config.attack_type.value,
        "malicious_ids": malicious_ids,
        "all_agent_ids": all_agent_ids,
        "detector_ids": detector_ids,
        "task_id": task["task_id"],
        "task_signature": _task_signature(task),
        "question": task.get("question", ""),
        "gold_answer": task["answer"],
        "attack_target_answer": task.get("attack_target_answer", ""),
        "attack_target_option": task.get("attack_target_option", ""),
        "attack_success_metric": task.get("attack_success_metric", ""),
        "output_mode": task.get("output_mode", ""),
        "entry_point": task.get("entry_point", ""),
        "test": task.get("test", ""),
        "dataset_name": task.get("dataset_name", ""),
        "score_totals": task.get("score_totals", {}),
        "num_private_shards": task.get("num_private_shards", ""),
        "rep": exp_config.rep,
        "decision_protocol": "system_majority_vote",
        "coordination_structure": "public_peer_discussion",
        **_runtime_context(),
        **_condition_labels(exp_config, malicious_count),
    }

    sim = Simulation(workers, detectors, cfg.TOTAL_TURNS, experiment_id, metadata, exp_config.domain)
    await sim.run()

    evaluator = Evaluator(sim.logger.db_path, exp_config.domain)
    result = {
        "unsafe_output_rate": evaluator.compute_unsafe_output_rate(),
        "functional_correctness": evaluator.compute_functional_correctness(),
        "completion_failure": evaluator.compute_completion_failure(),
        "attack_success": evaluator.compute_attack_success(),
        "false_positive_rate": evaluator.compute_false_positive_rate(),
        "detection_rate": evaluator.compute_detection_rate(),
        "detection_precision": evaluator.compute_detection_precision(),
        "attribution_accuracy": evaluator.compute_attribution_accuracy(),
        "detection_latency": evaluator.compute_detection_latency(),
        "benign_restriction_rate": evaluator.compute_benign_restriction_rate(),
        "benign_recovery_time": evaluator.compute_benign_recovery_time(),
        "wall_time_seconds": time.time() - started,
        "experiment_id": experiment_id,
        "condition_id": exp_config.condition_id,
        "condition_name": exp_config.condition_name,
        "attack_type": exp_config.attack_type.value,
        "m": exp_config.m,
        "malicious_count": malicious_count,
        "d": exp_config.d,
        "rep": exp_config.rep,
        "task_id": task["task_id"],
        "gold_answer": task["answer"],
        "attack_target_answer": task.get("attack_target_answer", ""),
        "attack_target_option": task.get("attack_target_option", ""),
        "_detection_stats_valid": evaluator.detection_stats_valid(),
    }
    evaluator.close()
    return result


async def run_single_experiment(
    exp_config: ExperimentConfig,
    task_pools: Dict[str, List[Dict[str, Any]]],
    task_pool_signatures: Dict[str, str],
    task_progress: Dict[str, Dict[str, Any]],
    progress_lock: asyncio.Lock,
) -> Dict[str, Any]:
    tasks = task_pools[exp_config.domain]
    task_pool_signature = task_pool_signatures[exp_config.domain]
    results: List[Dict[str, Any]] = []
    resumed_tasks = 0
    fresh_tasks = 0

    for task_index, task in enumerate(tasks):
        key = _task_key(exp_config, task, task_pool_signature)
        if key in task_progress:
            results.append(task_progress[key])
            resumed_tasks += 1
            continue

        result = await _run_single_task(exp_config, task, task_index)
        results.append(result)
        task_progress[key] = result
        fresh_tasks += 1
        await _append_task_progress(TASK_PROGRESS_JSONL, progress_lock, key, result)
        print(
            f"Task done: {exp_config.condition_id} rep={exp_config.rep} "
            f"task={task_index + 1}/{len(tasks)} acc={result['functional_correctness']} "
            f"attack_success={result['attack_success']} elapsed={result['wall_time_seconds']:.1f}s"
        )

    if resumed_tasks > 0:
        print(f"Resumed {resumed_tasks} task results and ran {fresh_tasks} new tasks for {exp_config.condition_id} rep={exp_config.rep}")

    aggregate = _aggregate(results)
    aggregate["task_pool_signature"] = task_pool_signature
    return aggregate


def _estimate_eta_seconds(pending_task_count: int, max_concurrent: int) -> float:
    return pending_task_count * cfg.ESTIMATED_SECONDS_PER_TASK / max(1, max_concurrent)


def _maybe_adjust_concurrency_for_eta(pending_task_count: int, max_concurrent: int) -> int:
    eta_limit = cfg.FULL_RUN_ETA_LIMIT_DAYS * 24 * 3600
    eta = _estimate_eta_seconds(pending_task_count, max_concurrent)
    print(
        f"ETA estimate: pending_tasks={pending_task_count}, seconds_per_task={cfg.ESTIMATED_SECONDS_PER_TASK:.1f}, "
        f"concurrency={max_concurrent}, wall_eta={eta / 3600:.2f}h ({eta / 86400:.2f}d)."
    )
    if eta <= eta_limit or not cfg.AUTO_RAISE_CONCURRENCY_FOR_ETA:
        return max_concurrent
    needed = math.ceil((pending_task_count * cfg.ESTIMATED_SECONDS_PER_TASK) / eta_limit)
    adjusted = min(MAX_CONCURRENT_CAP, max(max_concurrent, needed))
    adjusted_eta = _estimate_eta_seconds(pending_task_count, adjusted)
    if adjusted != max_concurrent:
        print(f"ETA exceeded {cfg.FULL_RUN_ETA_LIMIT_DAYS:.1f} days; raising concurrency to {adjusted} (estimated {adjusted_eta / 86400:.2f}d).")
    if adjusted_eta > eta_limit:
        print(
            "WARNING: ETA still exceeds limit after concurrency adjustment. "
            "Use NUM_SHARDS/SHARD_INDEX by launching multiple tmux sessions, or raise RATE_LIMIT_PER_SEC if Azure quota permits."
        )
    return adjusted


async def run_all_experiments(configs: List[ExperimentConfig], max_concurrent: int) -> None:
    Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=True)
    csv_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()

    _validate_output_csv_header(OUTPUT_CSV)
    completed_configs = _load_completed_configs(OUTPUT_CSV)
    task_progress = _load_task_progress(TASK_PROGRESS_JSONL)
    task_pools = _build_task_pools(seed=cfg.GLOBAL_SEED)
    task_pool_signatures = {
        domain: _task_pool_signature(tasks)
        for domain, tasks in task_pools.items()
    }

    pending_configs = [
        config_item
        for config_item in configs
        if _config_key(config_item, task_pool_signatures[config_item.domain]) not in completed_configs
    ]
    pending_task_count = sum(
        1
        for config_item in pending_configs
        for task in task_pools[config_item.domain]
        if _task_key(config_item, task, task_pool_signatures[config_item.domain]) not in task_progress
    )
    max_concurrent = _maybe_adjust_concurrency_for_eta(pending_task_count, max_concurrent)

    print(
        f"Resume mode: completed_configs={len(completed_configs)}, pending_configs={len(pending_configs)}, "
        f"checkpointed_tasks={len(task_progress)}, output={OUTPUT_CSV}, progress={TASK_PROGRESS_JSONL}, "
        f"run_label={RUN_LABEL}, version={EXPERIMENT_VERSION}, concurrency={max_concurrent}."
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(exp_config: ExperimentConfig) -> None:
        async with semaphore:
            cfg_key = _config_key(exp_config, task_pool_signatures[exp_config.domain])
            started = time.time()
            if cfg_key in completed_configs:
                print("Skipping already-completed config:", exp_config.condition_id, "rep", exp_config.rep)
                return

            malicious_count = cfg.NUM_WORKERS if exp_config.m == "all" else int(exp_config.m)
            for attempt in range(1, CONFIG_RETRY_LIMIT + 1):
                try:
                    print(
                        f"Starting config: {exp_config.condition_id} rep={exp_config.rep} "
                        f"attempt={attempt}/{CONFIG_RETRY_LIMIT}"
                    )
                    result = await run_single_experiment(
                        exp_config,
                        task_pools,
                        task_pool_signatures,
                        task_progress,
                        progress_lock,
                    )
                    row = _row_for_output(
                        exp_config,
                        result,
                        malicious_count,
                        task_pool_signatures[exp_config.domain],
                    )
                    async with csv_lock:
                        if cfg_key in completed_configs:
                            return
                        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
                            if f.tell() == 0:
                                writer.writeheader()
                            writer.writerow(row)
                            f.flush()
                        completed_configs.add(cfg_key)
                    print(f"Completed config: {exp_config.condition_id} rep={exp_config.rep} in {time.time() - started:.1f}s")
                    return
                except Exception as exc:
                    if not _is_retryable_error(exc) or attempt == CONFIG_RETRY_LIMIT:
                        print(f"FAILED permanently: {exp_config} -> {exc}")
                        async with csv_lock:
                            with open(FAILED_CONFIGS_CSV, "a", encoding="utf-8") as failed:
                                failed.write(f"{exp_config},{exc}\n")
                        return
                    wait_s = _backoff_seconds(attempt)
                    print(f"Retryable error for {exp_config.condition_id}: {exc}. Sleeping {wait_s:.1f}s.")
                    await asyncio.sleep(wait_s)

    await asyncio.gather(*(run_with_semaphore(config_item) for config_item in pending_configs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run collusive covert experiments")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny validation matrix instead of the full experiment")
    parser.add_argument("--smoke-tasks", type=int, default=2, help="Task count used with --smoke")
    parser.add_argument("--tasks", type=int, default=None, help="Override tasks per condition")
    parser.add_argument("--task-ids-path", type=str, default=None, help="Use an explicit frozen task-id manifest instead of sampled tasks")
    parser.add_argument("--reps", type=int, default=None, help="Override repetitions")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Concurrent condition runners")
    parser.add_argument("--run-label", type=str, default=None, help="Override RUN_LABEL for outputs")
    parser.add_argument("--output-csv", type=str, default=None, help="Aggregate output CSV path")
    parser.add_argument("--task-progress-jsonl", type=str, default=None, help="Task checkpoint JSONL path")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated condition_name filter for targeted readiness/audit runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned configs and ETA without API calls")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle condition order")
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global RUN_LABEL, OUTPUT_CSV, TASK_PROGRESS_JSONL, TOTAL_REPS
    if args.run_label:
        RUN_LABEL = _slugify(args.run_label)
        if args.output_csv is None:
            OUTPUT_CSV = f"experiment_results.{RUN_LABEL}.csv"
        if args.task_progress_jsonl is None:
            TASK_PROGRESS_JSONL = f"task_progress.{RUN_LABEL}.jsonl"
    if args.output_csv:
        OUTPUT_CSV = args.output_csv
    if args.task_progress_jsonl:
        TASK_PROGRESS_JSONL = args.task_progress_jsonl
    if args.smoke:
        cfg.QA_TASKS = int(args.smoke_tasks)
        TOTAL_REPS = 1
    if args.tasks is not None:
        cfg.QA_TASKS = int(args.tasks)
    if args.task_ids_path is not None:
        cfg.TASK_POOL_IDS_PATH = args.task_ids_path
    if args.reps is not None:
        TOTAL_REPS = int(args.reps)


def _print_smoke_summary(output_csv: str) -> None:
    if not os.path.exists(output_csv):
        return
    rows: List[Dict[str, str]] = []
    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    print("\nSmoke summary:")
    for row in rows:
        print(
            f"- {row['condition_id']} rep={row['rep']}: "
            f"accuracy={row.get('functional_correctness_mean')} "
            f"attack_success={row.get('attack_success_mean')} "
            f"completion_failure={row.get('completion_failure_mean')}"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    args = parse_args()
    _apply_cli_overrides(args)

    if cfg.GLOBAL_SEED is not None:
        random.seed(cfg.GLOBAL_SEED)

    if args.smoke:
        configs = generate_smoke_configs()
    else:
        configs = generate_configs(TOTAL_REPS)
    if args.conditions:
        requested_conditions = {
            condition.strip()
            for condition in args.conditions.split(",")
            if condition.strip()
        }
        known_conditions = {config_item.condition_name for config_item in configs}
        unknown_conditions = requested_conditions - known_conditions
        if unknown_conditions:
            raise ValueError(
                "Unknown condition(s): "
                + ", ".join(sorted(unknown_conditions))
                + ". Known conditions: "
                + ", ".join(sorted(known_conditions))
            )
        configs = [
            config_item
            for config_item in configs
            if config_item.condition_name in requested_conditions
        ]
    if not args.no_shuffle:
        random.shuffle(configs)

    tasks_per_config = cfg.QA_TASKS
    total_task_runs = len(configs) * tasks_per_config
    condition_names = sorted({config_item.condition_name for config_item in configs})
    print("=== Collusive Covert Experiment ===")
    print(f"domain={DOMAIN} configs={len(configs)} tasks_per_config={tasks_per_config} total_task_runs={total_task_runs}")
    print(f"turns={cfg.TOTAL_TURNS} max_tokens={cfg.MAX_TOKENS} model={cfg.MODEL_NAME} output={OUTPUT_CSV}")
    print(f"attack_type={FIXED_ATTACK_TYPE.value}")
    print(f"conditions={condition_names}")

    if args.dry_run:
        _maybe_adjust_concurrency_for_eta(total_task_runs, args.max_concurrent)
        for config_item in configs[:20]:
            print(config_item)
        if len(configs) > 20:
            print(f"... {len(configs) - 20} more configs")
        return

    asyncio.run(run_all_experiments(configs, args.max_concurrent))
    if args.smoke:
        _print_smoke_summary(OUTPUT_CSV)
    print(f"Done. Aggregate results saved to {OUTPUT_CSV}; task checkpoints saved to {TASK_PROGRESS_JSONL}.")


if __name__ == "__main__":
    main()
