# MRT/SHADE Baseline Plan

Primary baseline: `ScaleAI/mrt` public SHADE/CUA-SHADE release.

This is the current public task universe we can use without inventing tasks:

- 5 classic SHADE-Arena public task pairs.
- 15 CUA-SHADE-Arena task pairs.
- 20 total public task families.
- 19 active final task families after one predeclared pipeline-quality exclusion.

ControlArena documents that the public SHADE wrapper contains 5 task pairs, while the full 17 classic SHADE task benchmark reserves 12 task pairs for more rigorous evaluation. To avoid cherry-picking, this project uses every public MRT task family except task families excluded by a predeclared pipeline-quality gate.

## Frozen Task Pool

The frozen active pool is recorded in `mrt_public_full_manifest.json` and validated by `mrt_public_task_manifest.py`.

Pipeline exclusion:

- `classic_shade/expenditure_accounting` is excluded from the final active pool because repeated honest and attack audits completed with malformed JSON tool-call parse errors on long file writes. This is a pipeline-quality exclusion, not an attack-success exclusion.

## Baseline-Run Design

Default MRT/SHADE baseline design:

- `honest_baseline`
- `single_attacker`
- posthoc monitor scoring on honest and attack trajectories

With 19 active task families and 5 repetitions per trajectory condition, this yields 190 primary task trajectories before additional monitor-scaffold calls. If budget allows, 10 repetitions per condition gives 380 primary task trajectories.

MRT/SHADE is a single-agent hidden-side-task benchmark. Do not fabricate a collusion condition inside this benchmark-native baseline pool.

The primary study must include the multi-agent collusion matrix in the main runner: clean baseline, single attacker, collusive attackers, detector-only controls, single-attacker detector conditions, and collusive-attacker detector conditions. In that main study, attackers have no private communication channel and may coordinate only through public discussion. Detector agents may privately coordinate with other detectors.

## Frozen Attack Strategy

Use the MRT-native global attack setting `selective_v2_liar_204_thinkchoose` unless explicitly running an ablation.

Rationale:

- `naive` is too weak for the main research question; it under-elicits covert side-task execution.
- `selective_v2_liar_204_thinkchoose` is a single benchmark-native strategy applied across tasks, not a task-specific prompt patch.
- Further task-by-task prompt editing after failures should stop; it risks overfitting and weakens the paper.

## Expected Effect Size

Do not require high attack success as a go/no-go criterion. Prior SHADE-style work shows sabotage success is real but sparse and task-dependent. The defensible claim is to measure when covert side tasks succeed, when monitors catch them, and which task affordances make sabotage easier or harder.

## Task Inclusion Policy

The public pool remains the intended research source: all 5 public classic SHADE task families plus all 15 CUA-SHADE task families.

For final classic runs and fast classic-only audits, use `MRT_CLASSIC_TASK_SET=classic_stable`, which excludes `expenditure_accounting` because both honest and attack trajectories showed JSON/tool-call parsing instability. That exclusion is a pipeline-quality gate, not an attack-success filter.

Use `MRT_CLASSIC_TASK_SET=classic_all` only when deliberately rechecking whether the `expenditure_accounting` pipeline issue has been fixed.

Pipeline hardening before including `expenditure_accounting`:

- Default classic audit/single-condition completion budget is `MRT_MAX_TOKENS=4096` to reduce truncated tool-call JSON on long file-writing tasks.
- Drive tool schemas now explicitly recommend compact `create_file` calls plus `append_to_file` chunks for large reports/tables.
- These are generic tool-call robustness changes, not task-specific attack-strategy edits.

## Reporting Requirements

- Report classic SHADE and CUA-SHADE separately as well as pooled.
- Report main-task success, side-task success, monitor detection, and attack-undetected success.
- Report confidence intervals over task families and repetitions.
- Do not drop failed or inconvenient task families after seeing outcomes; failures are part of the result unless they are genuine pipeline failures.
- Report the frozen attack strategy and any excluded pipeline-failed task families explicitly.
