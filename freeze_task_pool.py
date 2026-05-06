"""Write the exact QA task ids selected by the current seed/settings to a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as cfg
from qa_loader import load_qa_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current QA task pool to a manifest")
    parser.add_argument("--tasks", type=int, default=cfg.QA_TASKS, help="Number of tasks to freeze")
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED, help="Seed used for task sampling")
    parser.add_argument("--output", type=str, required=True, help="Path to write JSON task-id manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_qa_tasks(num_tasks=args.tasks, seed=args.seed)
    task_ids = [task["task_id"] for task in tasks]
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(task_ids, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(task_ids)} task ids to {output_path}")


if __name__ == "__main__":
    main()
