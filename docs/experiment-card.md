# Experiment card: SO-101 generalisation

What was fixed before any result was seen, exactly which models and data were used, every budget including what failed,
and what remains unmeasured. Results and reasoning: [`report.pdf`](report.pdf) ([source](report.tex)) and
[`generalisation-diagnosis.md`](generalisation-diagnosis.md).

## Frozen before scoring

| Item | Value |
|---|---|
| Task | Pick the named object, place it on the named target (SO-101, MuJoCo twin) |
| Benchmark | Six suites × 12 scenes × 8 perturbed rehearsals = 96 each, 576 in all, seed 42 ([`benchmark.py`](../src/nebius_physical_ai/benchmark.py)) |
| Success | `StackCheck`: pick block resting on the target's upper face ([`rollout.py`](../src/nebius_physical_ai/twin/rollout.py)) |
| Comparison | Paired McNemar per suite, on scenes where the two checkpoints disagree |
| Promotion gate | Pooled paired gain and no suite significantly worse ([`benchmark.promotion`](../src/nebius_physical_ai/benchmark.py)); fixed before DAgger round 1 was scored |
| Baseline | v5, scored once and reused as the reference for every comparison |
| External check | so101-benchmark `stack_blue_on_red`, 50 episodes |

## Models, checkpoints and versions

| Role | Model / checkpoint | Version |
|---|---|---|
| Base policy | [`hqfang/groot-so100_101`](https://huggingface.co/hqfang/groot-so100_101) (GR00T N1.7, 150k steps) | commit `f5e17af`; `model.safetensors` sha256 `9ef04e19…` |
| Training / serving | LeRobot | 0.6.1 (`lerobot[groot,training]`) |
| v5 | base + 800 v4c twin demos, 12k steps of 8, lr 2.5e-5 | `gr00t-preflight-v5/checkpoints/012000` |
| DAgger round 2 (promoted) | v5 + 4k steps at lr 1e-5 on 2,400 base demos + 157 recoveries ×8 | `gr00t-preflight-dagger-r2/checkpoints/004000` |
| FLUX 3 Action | [`black-forest-labs/flux-3-action-so101`](https://huggingface.co/black-forest-labs/flux-3-action-so101) + LoRA rank 32 | LeRobot main `ca69a20`; 24k microsteps (12k updates of 8) |
| Simulator | MuJoCo, menagerie `robotstudio_so101` | 3.14.0 |
| RL expert | stable-baselines3 PPO, CPU PyTorch | 2.9.0, 2.14.1+cpu |

## Budgets, failures included

| Campaign | Attempted | Kept / succeeded | Not used |
|---|---|---|---|
| DAgger r1 harvest (v5) | 800 rollouts | 456 succeeded; 207 of 330 recoverable failures kept; 85 recovered | 123 failures over budget; 122 kept failures the expert could not recover; 14 *holding* not recoverable |
| DAgger r2 harvest (v5) | 800 | 494; 189 of 296 kept; 72 recovered | 107 over budget; 117 unrecovered; 10 *holding* |
| DAgger r3 harvest (round 2) | 800 | 574; 146 of 197 kept; 59 recovered | 51 over budget; 87 unrecovered; 29 *holding* |
| Promotion attempts | 2 rounds scored | 1 promoted (round 2) | round 1 kept, not promoted |
| FLUX | zero-shot, LoRA v8, fair LoRA, 16-step | 0 promoted | all four worse than v5 |
| DAgger r3 retrain | queued | — | first r3 checkpoint deleted by the round script on KEEP |
| Best-of-8 | 2 suites × 96 on round 2 | 142 succeeded | within noise; not adopted |
| RL expert | plain PPO (stopped at 4.4M steps), 2 relaunches killed for bugs | residual PPO: 71.9% held-out snapshot recovery at 2.5M steps; shaped PPO ~6% training success | runs tagged `superseded` in W&B |
| ArmnetBench conversion | 400 teleop + 515 successful rollouts selected from 2,499 episodes | all 915 converted | 1,532 failed and 52 suboptimal rollouts excluded by design |

## Filters: before or after seeing results

- **Before:** the six suites, the seed, the gate, the uncertainty band (scene success in 5–80%), the per-round recording
  budget, and the decision to exclude *holding* from recoveries.
- **After, and declared:**
  - Upweighting recoveries ×8 in round 2, chosen after round 1 was flat.
  - Stopping FLUX's so101-benchmark at 0/7 to free the GPU.
  - The RL expert's potential-based shaping, after the plain reward was found to pay hovering over stacking.

## Not measured

- Anything on the physical arm (not delivered).
- More than one training seed per checkpoint, or more than one benchmark seed.
- Whether FLUX's closed-loop failure comes from its 8-frame history or its delta actions (hypothesis, untested).
- New-object generalisation from real data (v9 co-training running).
- Whether round 3's held_out_objects gain (p=0.043, one suite of six, uncorrected) replicates on a retrain.
