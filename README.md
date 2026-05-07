# Covert Collusive LLMs

Research code for the coordination-collusion experiment described in *Eyes in the Backchannel: Detecting Covert Collusion in LLM Multi-Agent Systems*.

This repository intentionally contains the final benchmark code only. Earlier exploratory branches for HotpotQA, MuSiQue, MMLU-Pro, HumanEval, BigCodeBench, APPS-Control, MRT/SHADE, and one-off smoke tests were removed so the public repo reflects the experiment actually reported in the paper.

## What This Benchmark Tests

The final experiment studies covert influence in a multi-agent system without giving any worker control over the final answer.

Each task presents the same candidate options to five agents. Each agent privately receives different numeric evidence about those options. The correct answer is the option with the highest total score after combining all agents' private information.

Agents deliberate publicly, then submit private final votes. The system aggregates those votes. This removes the earlier designated-reporter confound: attackers must influence the group decision rather than simply write the final response.

## Conditions

The full matrix is:

- `clean_baseline`
- `clean_one_detector`
- `clean_two_detectors`
- `single_attacker`
- `single_attacker_one_detector`
- `single_attacker_two_detectors`
- `collusive_attack`
- `collusive_attack_one_detector`
- `collusive_attack_two_detectors`

Attackers do not have a private communication channel. Detector agents do have a detector-only private channel, and can recommend temporary quarantine/restriction through the permission manager. Quarantine and recovery are preserved; prior messages are not removed from the transcript.

## Repository Layout

- `coordination_collusion_loader.py` defines the synthetic coordination-collusion task pool.
- `parallel_experiment_runner.py` runs the 9-condition experiment matrix.
- `agent.py`, `detector_agent.py`, `simulation.py`, `permission_manager.py`, and `evaluation.py` implement workers, detectors, the dialogue loop, interventions, and metrics.
- `task_pool_ids.coordination_collusion.seed0.audit10.json` is the frozen audit manifest.
- `task_pool_ids.coordination_collusion.seed0.full25_holdout.json` is the frozen final-run holdout manifest.
- `paper/` contains the revised manuscript source.

Generated logs, SQLite databases, CSV/JSONL outputs, local credentials, virtual environments, caches, and exploratory smoke-test artifacts are intentionally ignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.mlsecurity2
```

Fill in `.env.mlsecurity2` locally. Do not commit API keys, private keys, run logs, or result databases.

## Dry Run

```bash
TASK_DOMAIN=knowledge_qa \
QA_DATASET_NAME=coordination_collusion \
QA_TASKS=10 \
NUM_WORKERS=5 \
TOTAL_TURNS=6 \
MAX_TOKENS=700 \
TASK_POOL_IDS_PATH=task_pool_ids.coordination_collusion.seed0.audit10.json \
python parallel_experiment_runner.py \
  --tasks 10 \
  --task-ids-path task_pool_ids.coordination_collusion.seed0.audit10.json \
  --reps 1 \
  --max-concurrent 4 \
  --conditions clean_baseline,clean_one_detector,clean_two_detectors,single_attacker,single_attacker_one_detector,single_attacker_two_detectors,collusive_attack,collusive_attack_one_detector,collusive_attack_two_detectors \
  --dry-run
```

## Final-Run Configuration

The reported full run used:

- `QA_DATASET_NAME=coordination_collusion`
- `TASK_POOL_IDS_PATH=task_pool_ids.coordination_collusion.seed0.full25_holdout.json`
- 25 holdout tasks
- 5 repetitions
- 9 conditions
- 5 agents
- 6 discussion turns
- Azure OpenAI `gpt-5.2`

The run produced 1,125 task-condition executions with zero completion failures.

## Reproducibility Notes

The runner records protocol fingerprints, task-pool signatures, condition metadata, task-level checkpoints, and aggregate metrics. Interrupted runs can be resumed with the same command and run label.

Task selection and prompt edits should be frozen before any reported run. Do not remove or add tasks based on attack success or detector performance.
