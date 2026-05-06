"""
Generate paper-ready tables and figures from QA-only experiment results.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = os.getenv("INPUT_CSV", "experiment_results.csv")
OUTPUT_TABLE_DIR = Path(os.getenv("OUTPUT_TABLE_DIR", "results/paper_tables"))
OUTPUT_FIG_DIR = Path(os.getenv("OUTPUT_FIG_DIR", "results/paper_figures"))
EXPECTED_TASKS = int(os.getenv("EXPECTED_TASKS", "25"))
REQUIRE_FULL_TASK_COUNTS = os.getenv("REQUIRE_FULL_TASK_COUNTS", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

METRIC_COLUMNS = [
    "functional_correctness_mean",
    "false_positive_rate_mean",
    "detection_rate_mean",
    "detection_latency_mean",
    "benign_restriction_rate_mean",
    "benign_recovery_time_mean",
]

STD_COLUMNS = [
    "functional_correctness_std",
    "false_positive_rate_std",
    "detection_rate_std",
    "detection_latency_std",
    "benign_restriction_rate_std",
    "benign_recovery_time_std",
]


def ensure_dirs() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
def _format_mean_std(mean_series: pd.Series, std_series: pd.Series, digits: int = 3) -> pd.Series:
    return mean_series.round(digits).astype(str) + " ± " + std_series.round(digits).astype(str)


def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find results file: {path}")
    df = pd.read_csv(path)
    required = {
        "domain",
        "m",
        "d",
        "condition_name",
        "tasks_evaluated",
        "functional_correctness_mean",
        "false_positive_rate_mean",
        "detection_rate_mean",
        "detection_latency_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in set(METRIC_COLUMNS + STD_COLUMNS + ["m", "d", "tasks_evaluated"]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["domain"].astype(str) == "knowledge_qa"]
    if REQUIRE_FULL_TASK_COUNTS:
        df = df[df["tasks_evaluated"] == EXPECTED_TASKS]
    return df.reset_index(drop=True)


def save_summary(df: pd.DataFrame) -> None:
    grouped = df[METRIC_COLUMNS + STD_COLUMNS].mean(numeric_only=True).to_frame().T
    grouped["functional_correctness"] = _format_mean_std(grouped["functional_correctness_mean"], grouped["functional_correctness_std"])
    grouped["false_positive_rate"] = _format_mean_std(grouped["false_positive_rate_mean"], grouped["false_positive_rate_std"])
    grouped["detection_rate"] = _format_mean_std(grouped["detection_rate_mean"], grouped["detection_rate_std"])
    grouped[["functional_correctness", "false_positive_rate", "detection_rate"]].to_csv(
        OUTPUT_TABLE_DIR / "summary.csv", index=False
    )


def save_condition_table(df: pd.DataFrame) -> None:
    out = df.copy()
    out = out.sort_values(by=["m", "d", "rep"], ascending=[True, True, True])
    out.to_csv(OUTPUT_TABLE_DIR / "condition_table.csv", index=False)


def plot_metric_by_condition(df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    metric_col = f"{metric}_mean"
    if metric_col not in df.columns:
        return
    subset = df.copy()
    subset["condition"] = subset["condition_name"].astype(str)
    grouped = subset.groupby("condition", dropna=False)[metric_col].mean().reset_index()
    plt.figure(figsize=(14, 6))
    plt.bar(grouped["condition"], grouped[metric_col])
    plt.xticks(rotation=70, ha="right")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_results(INPUT_CSV)
    save_summary(df)
    save_condition_table(df)
    plot_metric_by_condition(df, "detection_rate", "Detection Rate by Condition", "detection_rate_by_condition.png")
    plot_metric_by_condition(df, "functional_correctness", "Functional Correctness by Condition", "functional_correctness_by_condition.png")
    print(f"Saved paper tables to {OUTPUT_TABLE_DIR.resolve()}")
    print(f"Saved paper figures to {OUTPUT_FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
