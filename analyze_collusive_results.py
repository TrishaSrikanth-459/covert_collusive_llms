"""
Statistical analysis for coordination-collusion experiment checkpoints.

Reads task_progress.<run_label>.jsonl and writes pairwise tests for the core
research comparisons specified in the experiment plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback for minimal environments
    stats = None
    HAS_SCIPY = False


COMPARISONS = [
    ("collusion_impact", "collusive_attack", "single_attacker"),
    ("attack_effectiveness", "collusive_attack", "clean_baseline"),
    ("clean_one_detector_effect", "clean_one_detector", "clean_baseline"),
    ("clean_two_detector_benefit", "clean_two_detectors", "clean_one_detector"),
    ("single_one_detector_benefit", "single_attacker_one_detector", "single_attacker"),
    ("single_two_detector_benefit", "single_attacker_two_detectors", "single_attacker_one_detector"),
    ("collusive_one_detector_benefit", "collusive_attack_one_detector", "collusive_attack"),
    ("collusive_two_detector_benefit", "collusive_attack_two_detectors", "collusive_attack_one_detector"),
]
BINARY_METRICS = [
    "functional_correctness",
    "completion_failure",
    "attack_success",
    "detection_rate",
    "false_positive_rate",
    "benign_restriction_rate",
]
COUNT_METRICS = ["detection_latency", "wall_time_seconds"]


def _load_task_results(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("task_key")
            result = rec.get("result")
            if not key or not isinstance(result, dict):
                continue
            # Last write wins for duplicate checkpoints.
            if key in seen:
                rows = [row for row in rows if row.get("_task_key") != key]
            result = dict(result)
            result["_task_key"] = key
            rows.append(result)
            seen.add(key)
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _values(rows: Iterable[Dict[str, Any]], metric: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        parsed = _safe_float(row.get(metric))
        if parsed is not None:
            vals.append(parsed)
    return vals


def _normal_approx_p(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return math.nan
    p1 = mean(a)
    p2 = mean(b)
    se = math.sqrt(max(1e-12, p1 * (1 - p1) / len(a) + p2 * (1 - p2) / len(b)))
    z = (p1 - p2) / se
    # two-sided normal tail via erf
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _binary_test(a: List[float], b: List[float]) -> Tuple[float, float, float, float]:
    if not a or not b:
        return math.nan, math.nan, math.nan, math.nan
    p_a = mean(a)
    p_b = mean(b)
    diff = p_a - p_b
    se = math.sqrt(max(1e-12, p_a * (1 - p_a) / len(a) + p_b * (1 - p_b) / len(b)))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    if HAS_SCIPY:
        table = [
            [sum(1 for x in a if x >= 0.5), sum(1 for x in a if x < 0.5)],
            [sum(1 for x in b if x >= 0.5), sum(1 for x in b if x < 0.5)],
        ]
        try:
            p_value = float(stats.fisher_exact(table).pvalue)
        except Exception:
            p_value = _normal_approx_p(a, b)
    else:
        p_value = _normal_approx_p(a, b)
    return diff, ci_low, ci_high, p_value


def _cliffs_delta(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return math.nan
    greater = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (len(a) * len(b))


def _continuous_test(a: List[float], b: List[float]) -> Tuple[float, float, float, float]:
    if not a or not b:
        return math.nan, math.nan, math.nan, math.nan
    diff = mean(a) - mean(b)
    # Bootstrap-style normal CI approximation with independent samples.
    var_a = sum((x - mean(a)) ** 2 for x in a) / max(1, len(a) - 1)
    var_b = sum((x - mean(b)) ** 2 for x in b) / max(1, len(b) - 1)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    if HAS_SCIPY:
        try:
            p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            p_value = math.nan
    else:
        p_value = math.nan
    return diff, ci_low, ci_high, p_value


def analyze(input_jsonl: str, output_csv: str) -> None:
    rows = _load_task_results(input_jsonl)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("attack_type")), str(row.get("condition_name")))].append(row)

    out_rows: List[Dict[str, Any]] = []
    attack_types = sorted({str(row.get("attack_type")) for row in rows if row.get("attack_type")})
    for attack_type in attack_types:
        for comparison_name, condition_a, condition_b in COMPARISONS:
            rows_a = grouped.get((attack_type, condition_a), [])
            rows_b = grouped.get((attack_type, condition_b), [])
            for metric in BINARY_METRICS:
                a = _values(rows_a, metric)
                b = _values(rows_b, metric)
                diff, ci_low, ci_high, p_value = _binary_test(a, b)
                out_rows.append({
                    "attack_type": attack_type,
                    "comparison": comparison_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "metric": metric,
                    "test": "fisher_exact" if HAS_SCIPY else "normal_approx",
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": mean(a) if a else math.nan,
                    "mean_b": mean(b) if b else math.nan,
                    "effect_size": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                })
            for metric in COUNT_METRICS:
                a = _values(rows_a, metric)
                b = _values(rows_b, metric)
                diff, ci_low, ci_high, p_value = _continuous_test(a, b)
                out_rows.append({
                    "attack_type": attack_type,
                    "comparison": comparison_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "metric": metric,
                    "test": "mann_whitney_u" if HAS_SCIPY else "mean_diff_only",
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": mean(a) if a else math.nan,
                    "mean_b": mean(b) if b else math.nan,
                    "effect_size": _cliffs_delta(a, b),
                    "mean_difference": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                })

    fieldnames = sorted({key for row in out_rows for key in row.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} statistical rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coordination-collusion task checkpoints")
    parser.add_argument("input_jsonl", help="task_progress.<run_label>.jsonl")
    parser.add_argument("--output", default="collusive_statistical_tests.csv")
    args = parser.parse_args()
    analyze(args.input_jsonl, args.output)


if __name__ == "__main__":
    main()
