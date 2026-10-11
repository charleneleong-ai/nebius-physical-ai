## Highlights

**Status, 11 Oct 2026 15:30 UTC.** DAgger round 2 is the served policy. The round 3 retrain is running; v9 real-data co-training was discarded. Everything is simulation; the physical SO-101 has not arrived.

| | Pooled (of 576) | held_out_objects (of 96) | so101-benchmark (of 50) |
|---|---|---|---|
| **Current best (served): DAgger round 2** | **464** | 50 | not run |
| Best unpromoted: DAgger round 3 | 481 (within noise of 464) | **61** (p=0.043 vs 50) | not run |
| First entry: v5 | 438 | 52 | 0 |

- **Recoveries from the policy's own failure states are the only lever that moved the gate**: round 2 (+26 pooled over v5) and round 3 (+17 more, held_out_objects +11).
- **Upweighting matters more than volume**: round 1's recoveries were 3.5% of training and did nothing; ×8 (about a third) made round 2.
- **Swapping the model or adding sampling did not help**: FLUX 3 Action LoRA 50/576; best-of-8 consensus within noise.
- **A residual RL expert recovers 74.0% of failure snapshots vs the scripted expert's 45.9%**, so it can feed DAgger states the script cannot recover.

**Pooled progression (gate suites, 6 × 96)**

| Date | Run | What changed | Pooled | held_out_objects |
|---|---|---|---|---|
| 06 Oct | v5 | base + 800 scripted twin demos | 438 | 52 |
| 07 Oct | v6 | recovery demos + lookalike decoys | 421 | 54 |
| 08 Oct | DAgger r1 | recoveries 3.5% of training | 436 | 47 |
| 09 Oct | **DAgger r2** | recoveries ×8 | **464** | 50 |
| 10 Oct | FLUX LoRA | different model, same demos and volume | 50 | 3 |
| 11 Oct | DAgger r3 | third round from r2 (not promoted) | 481 | **61** |

Companion page: [Preflight Generalisation Ledger](https://claude.ai/artifact/JUihdgT9xtqYc8a4ieJPQU). Report and experiment card: [docs/report.pdf](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/docs/report.pdf), [docs/experiment-card.md](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/docs/experiment-card.md).

## Task

Pick the named object and place it on the named target with an SO-101 arm, in a MuJoCo twin. Success is `StackCheck`: the pick block resting on the target's upper face. The metric is successes on six suites of 12 scenes × 8 perturbed rehearsals (96 each, 576 pooled, seed 42): held_out_objects, seen_vocabulary, clean_many, clean_top_down_many, real_1_pomade_tin_many, in_path_pomade_tin_many. Comparisons are paired McNemar tests on scenes where two checkpoints disagree. External check: so101-benchmark `stack_blue_on_red`, 50 episodes.

## Approach

The served policy is GR00T N1.7 (`hqfang/groot-so100_101`) fine-tuned with LeRobot 0.6.1: v5 (800 scripted twin demos, 12k steps, lr 2.5e-5), then DAgger round 2 (4k steps at lr 1e-5 on 2,400 base demos plus 157 recoveries ×8). Each DAgger round rolls the policy out on 800 rehearsals, snapshots every failure, keeps the most uncertain within a failure-type-balanced budget, and has the scripted expert record a recovery from each kept state ([`scripts/dagger_round.sh`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/scripts/dagger_round.sh), [`twin/dagger.py`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/src/nebius_physical_ai/twin/dagger.py)). Promotion needs the gate in [`benchmark.py`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/src/nebius_physical_ai/benchmark.py).

## What we learned

| Lever | Measured effect |
|---|---|
| DAgger recoveries ×8 (round 2) | +26 pooled vs v5, real_1_pomade_tin +12 (p=0.004) |
| A further DAgger round (round 3) | +17 pooled vs round 2 (within noise), held_out_objects +11 (p=0.043) |
| Residual RL on the scripted expert | snapshot recovery 45.9% → 74.0% |

**Closed / do not retry**

- FLUX 3 Action LoRA: predicts well open-loop (2.71 vs 8.24 hold-pose error at 32 steps) but 50/576 closed-loop; 16-step chunks did not fix it.
- Best-of-8 chunk medoid: within noise at 8× inference.
- Targeted demos (v6): staged mistakes and decoys did not move any suite.
- Visual domain randomisation (v7): fixes the wrist-mount scenes but costs clean_many 23 successes.
- Plain PPO from scratch on privileged state: reward paid hovering; 1% recovery.
- Real-data co-training (v9, 400 ArmnetBench demos at 23%): 373 vs 438, no gain on new objects, worse on three suites.

## Next

- Running: DAgger round 3 retrain, to check whether held_out_objects 61 replicates. v9b (v9 plus 515 real rollouts) is queued pending a decision, since v9 regressed.
- Queued: DAgger round 4 with the residual RL expert recording recoveries; the scripted expert recovers 40% of kept states, the RL expert about 72%.
- Deadline: hackathon submission 30 Oct 2026 17:00 UTC. The physical arm has not arrived; the repo will be updated with live results when it does.
