## Highlights

**Status, 12 Oct 2026 08:45 UTC.** DAgger round 3 (retrained) is the served policy. Round 4 with RL-expert recoveries was kept, not promoted; v9 real-data co-training was discarded. Everything is simulation; the physical SO-101 has not arrived.

| | Pooled (of 576) | held_out_objects (of 96) | so101-benchmark (of 50) |
|---|---|---|---|
| **Current best (served): DAgger round 3, retrained** | **483** | 57 | not run |
| Best on new objects: DAgger round 3, first run | 481 | **61** (p=0.043 vs 50) | not run |
| First entry: v5 | 438 | 52 | 0 |

- **Recoveries from the policy's own failure states are the only lever that moved the gate**: round 2 (+26 pooled over v5) and round 3 (+19 more on the retrain, which is now served).
- **Upweighting matters more than volume**: round 1's recoveries were 3.5% of training and did nothing; ×8 (about a third) made round 2.
- **Swapping the model or adding sampling did not help**: FLUX 3 Action LoRA 50/576; best-of-8 consensus within noise.
- **A residual RL expert recovers 74.0% of failure snapshots vs the scripted expert's 45.9%**, so it can feed DAgger states the script cannot recover.

**Pooled progression (gate suites, 6 × 96)**

| Date | Run | What changed | Pooled | held_out_objects |
|---|---|---|---|---|
| 06 Oct | v5 | base + 800 scripted twin demos | 438 | 52 |
| 07 Oct | v6 | recovery demos + lookalike decoys | 421 | 54 |
| 08 Oct | DAgger r1 | recoveries 3.5% of training | 436 | 47 |
| 09 Oct | DAgger r2 | recoveries ×8 | 464 | 50 |
| 10 Oct | FLUX LoRA | different model, same demos and volume | 50 | 3 |
| 11 Oct | DAgger r3 | third round from r2 (not promoted) | 481 | **61** |
| 11 Oct | v9 | + 400 real ArmnetBench demos | 373 | 48 |
| 12 Oct | **DAgger r3 retrain** | same recipe, retrained | **483** | 57 |
| 12 Oct | DAgger r4 | RL expert records recoveries | 471 | 49 |

Companion page: [Preflight Generalisation Ledger](https://claude.ai/artifact/JUihdgT9xtqYc8a4ieJPQU). Report and experiment card: [docs/report.pdf](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/docs/report.pdf), [docs/experiment-card.md](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/docs/experiment-card.md).

## Task

Pick the named object and place it on the named target with an SO-101 arm, in a MuJoCo twin. Success is `StackCheck`: the pick block resting on the target's upper face. The metric is successes on six suites of 12 scenes × 8 perturbed rehearsals (96 each, 576 pooled, seed 42): held_out_objects, seen_vocabulary, clean_many, clean_top_down_many, real_1_pomade_tin_many, in_path_pomade_tin_many. Comparisons are paired McNemar tests on scenes where two checkpoints disagree. External check: so101-benchmark `stack_blue_on_red`, 50 episodes.

## Approach

The served policy is GR00T N1.7 (`hqfang/groot-so100_101`) fine-tuned with LeRobot 0.6.1: v5 (800 scripted twin demos, 12k steps, lr 2.5e-5), then DAgger rounds 2 and 3 (each 4k steps at lr 1e-5 from the previous round, on 2,400 base demos plus all recoveries so far ×8; 216 by round 3). Each DAgger round rolls the policy out on 800 rehearsals, snapshots every failure, keeps the most uncertain within a failure-type-balanced budget, and has the scripted expert record a recovery from each kept state ([`scripts/dagger_round.sh`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/scripts/dagger_round.sh), [`twin/dagger.py`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/src/nebius_physical_ai/twin/dagger.py)). Promotion needs the gate in [`benchmark.py`](https://github.com/charleneleong-ai/nebius-physical-ai/blob/main/src/nebius_physical_ai/benchmark.py).

## What we learned

| Lever | Measured effect |
|---|---|
| DAgger recoveries ×8 (round 2) | +26 pooled vs v5, real_1_pomade_tin +12 (p=0.004) |
| A further DAgger round (round 3) | +17 / +19 pooled vs round 2 over two runs (the retrain passes the gate); held_out_objects +11 / +7 |
| RL-expert recoveries (round 4) | 88/120 recovered vs the script's ~40%, but 471 vs 483 (−12, within noise) |
| Residual RL on the scripted expert | snapshot recovery 45.9% → 74.0% |

**Closed / do not retry**

- FLUX 3 Action LoRA: predicts well open-loop (2.71 vs 8.24 hold-pose error at 32 steps) but 50/576 closed-loop; 16-step chunks did not fix it.
- Best-of-8 chunk medoid: within noise at 8× inference.
- Targeted demos (v6): staged mistakes and decoys did not move any suite.
- Visual domain randomisation (v7): fixes the wrist-mount scenes but costs clean_many 23 successes.
- Plain PPO from scratch on privileged state: reward paid hovering; 1% recovery.
- Real-data co-training (v9, 400 ArmnetBench demos at 23%): 373 vs 438, no gain on new objects, worse on three suites.

## Next

- Done overnight: the round 3 retrain (promoted, 483) and round 4 with RL-expert recoveries (kept, 471). v9b is cancelled.
- Next candidates: weight RL recoveries below ×8, or use the RL expert only for the failure types the script cannot fix (dropped, fell_off).
- Queued: DAgger round 4 with the residual RL expert recording recoveries; the scripted expert recovers 40% of kept states, the RL expert about 72%.
- Deadline: hackathon submission 30 Oct 2026 17:00 UTC. The physical arm has not arrived; the repo will be updated with live results when it does.
