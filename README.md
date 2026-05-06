# Covert Collusive LLMs

Research code for studying covert sabotage, collusion, and detection in LLM-based multi-agent systems.

The repository contains two complementary experiment tracks:

- A primary multi-agent coordination benchmark where benign workers each see only a private scoring shard, malicious workers try to steer the public group decision toward a hidden target, and trusted detector agents may privately coordinate with each other.
- An auxiliary MRT/SHADE baseline wrapper for public hidden-side-task sabotage tasks.

The current frozen readiness audit uses `QA_DATASET_NAME=coordination_collusion` with a fixed 9-condition matrix.

## Protocol

The primary experiment uses public peer discussion followed by private final votes. The system, not a worker-reporter, aggregates the final decision.

Attackers have no private communication channel. They can coordinate only through the public transcript visible to everyone.

Detector agents are trusted monitors. They can inspect the public transcript, use audit-only information where the task provides it, privately coordinate with other detector agents, and recommend temporary credibility restrictions or quarantine. Quarantine and recovery are preserved as part of the intervention model.

## Conditions

The full primary matrix is:

- `clean_baseline`
- `clean_one_detector`
- `clean_two_detectors`
- `single_attacker`
- `single_attacker_one_detector`
- `single_attacker_two_detectors`
- `collusive_attack`
- `collusive_attack_one_detector`
- `collusive_attack_two_detectors`

Core metrics include functional correctness, attack success, completion failure, false positive rate, detection rate, detection precision, attribution accuracy, benign restriction rate, and recovery behavior.

## Repository Layout

- `parallel_experiment_runner.py` runs the primary multi-agent matrix.
- `coordination_collusion_loader.py` defines the asymmetric-information collusion benchmark.
- `agent.py`, `detector_agent.py`, `simulation.py`, `permission_manager.py`, and `evaluation.py` implement the agent loop, detector loop, interventions, and metrics.
- `external_baselines/mrt/` vendors the public MRT/SHADE baseline code and task assets needed by the baseline scripts.
- `mrt_public_task_manifest.py` and `mrt_public_full_manifest.json` freeze the public MRT task pool and pipeline-quality exclusion policy.
- `run_mrt_final_pool.sh` launches the MRT/SHADE auxiliary baseline after explicit confirmation.
- `validate_mrt_final_readiness.py` validates the MRT/SHADE setup without making model calls.

Generated outputs are intentionally ignored: logs, SQLite DBs, task-progress JSONL files, result CSVs, local env files, virtual environments, IDE settings, and one-off smoke/probe artifacts.

Large AgentCompany fixture assets under `external_baselines/mrt/experiments/agentcompany/tac-openhands/` are intentionally not committed. Some fixture filenames and contents look like credentials even when used as benchmark data, so keep them out of the public repo and restore them from the trusted upstream MRT/AgentCompany source or the experiment VM before running CUA-SHADE locally.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local credential file from the template:

```bash
cp .env.example .env.mlsecurity2
```

Then fill in the values locally. Do not commit `.env`, `.env.mlsecurity2`, API keys, private keys, logs, or run outputs.

## Primary Multi-Agent Run

Example dry run for the frozen coordination-collusion matrix:

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

For a real run, source credentials first and remove `--dry-run`.

## MRT/SHADE Baseline

Validate the MRT/SHADE setup without model calls:

```bash
python validate_mrt_final_readiness.py
```

Inspect the final baseline launch plan:

```bash
./run_mrt_final_pool.sh --dry-run
```

The script refuses to start the expensive baseline unless `MRT_CONFIRM_FINAL_RUN=YES` is set.

CUA-SHADE runs also require the AgentCompany task asset directory mentioned above. The VM used for the experiment keeps those assets locally; this repository keeps only code, manifests, and lightweight task definitions.

## Reproducibility Notes

The code records protocol fingerprints, task-pool signatures, condition metadata, and task-level checkpoints. Interrupted runs can be resumed with the same command and run label.

Task exclusions should be made only for documented pipeline-quality failures, not because an attack fails or a detector underperforms. The current MRT/SHADE public-pool policy is documented in `MRT_BASELINE_PLAN.md`.
