## Highlights

**Status, 18 Oct 2026 16:00 AEDT.** DAgger round 3 (retrained) is still the served policy, now published as [chaleong/preflight-gr00t-so101-dagger-r3](https://huggingface.co/chaleong/preflight-gr00t-so101-dagger-r3). Experiments now run through **Autopilot**, a closed loop that turns Preflight's refusals and the benchmark's failures into one training cycle at a time: Nemotron on Token Factory chooses the recipe, and a paired gate (G3) judges it. Cycle 1 (multi-task, basket trained, ring held out) taught the new task, 6 → 86 of 96, transferred nothing to ring (0 → 0) and cost stacking (441 vs 483), so it was discarded. Cycle 4, a DAgger round answering four Preflight refusals, is training. Failure-state synthesis was discarded (437). Everything is simulation; the physical SO-101 has not arrived. Live: [Preflight](https://huggingface.co/spaces/chaleong/preflight) · [Autopilot](https://huggingface.co/spaces/chaleong/autopilot) · [Preflight Experiments](https://huggingface.co/spaces/chaleong/preflight-experiments).

| | Pooled (of 576) | held_out_objects (of 96) | so101-benchmark (of 50) |
|---|---|---|---|
| **Current best (served): DAgger round 3, retrained** | **483** | 57 | not run |
| Best on new objects: DAgger round 3, first run | 481 | **61** (p=0.043 vs 50) | not run |
| First entry: v5 | 438 | 52 | 0 |

- **Recoveries from the policy's own failure states are the only lever that moved the gate**: round 2 (+26 pooled over v5) and round 3 (+19 more on the retrain, which is now served).
- **Upweighting matters more than volume**: round 1's recoveries were 3.5% of training and did nothing; ×8 (about a third) made round 2.
- **Swapping the model or adding sampling did not help**: FLUX 3 Action LoRA 50/576; best-of-8 consensus and a trained DiVeR verifier both within noise.
- **A residual RL expert recovers 74.0% of failure snapshots vs the scripted expert's 45.9%**, but its recoveries did not help in DAgger (rounds 4-5).
- **A new task can be taught, task by task, at a price**: 400 demos took basket from 6 to 86 of 96, with no transfer to the held-out ring task and a 42-rehearsal stacking loss (Autopilot cycle 1).

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
| 13 Oct | DAgger r5 | RL recoveries weighted ×2 | 455 | 41 |
| 14 Oct | v11 | r3 + 87 calibrated real block-stacking episodes (9%) | 472 | 57 |
| 17 Oct | synthesis (r6) | expert recoveries from jittered failure states | 437 | 43 |
| 18 Oct | Autopilot cycle 1 | multi-task: + 400 basket demos, ring held out (basket 6 → 86) | 441 | 47 |

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
| RL-expert recoveries (rounds 4-5) | 73% / 64% of kept states recovered vs the script's ~40%, but 471 (×8) and 455 (×2) vs 483 |
| Residual RL on the scripted expert | snapshot recovery 45.9% → 74.0% |
| New-task demos (Autopilot cycle 1) | trained task basket 6 → 86/96; held-out ring 0 → 0; stacking −42 pooled |

**Closed / do not retry**

- FLUX 3 Action LoRA: predicts well open-loop (2.71 vs 8.24 hold-pose error at 32 steps) but 50/576 closed-loop; 16-step chunks did not fix it. Current-frame conditioning with absolute targets did, partly: held_out_objects 21 vs 3 at a quarter of the budget, still far below round 3.
- Best-of-8 chunk medoid: within noise at 8× inference.
- DiVeR verifier best-of-8 on round 3: step AUC 0.79 on held-out episodes, but 54 vs 57 and 85 vs 88. On the action expert's per-candidate state: 48 vs 57 (worse, p=0.049). With 305 real ArmnetBench outcomes: 52 vs 57.
- Targeted demos (v6): staged mistakes and decoys did not move any suite.
- Visual domain randomisation (v7): fixes the wrist-mount scenes but costs clean_many 23 successes.
- Plain PPO from scratch on privileged state: reward paid hovering; 1% recovery.
- Uncalibrated real-data co-training (v9, 400 ArmnetBench demos of eight tasks at 23%): 373 vs 438, worse on three suites. Calibrated and restricted to block stacking (v11) it is neutral: 472 vs 483.
- RL-expert recoveries in DAgger (rounds 4 and 5, at ×8 and ×2): 471 and 455 vs 483; new objects 49 then 41 vs 57.
- Failure-state synthesis (round 6): 437 vs 483, worse on three suites.
- Multi-task round at this mix (demos ×2, ~20% of episodes): teaches the trained task only and costs stacking. A gentler mix would be a new recipe.

## Next

- Running: Autopilot cycle 4, a DAgger round (round 7) chosen by Nemotron. It answers four Preflight refusals: the new-colour task, the real photo and two hard layouts. Judged by G3 on the objects and environments suites.
- Closed by the ledger, so Autopilot won't propose them: failure synthesis, multi-task at this mix, best-of-N, FLUX, v6, v7, RL-expert recoveries.
- Candidates for the menu: a gentler multi-task mix, wider camera jitter on the stacking data, a wrist camera.
- Freeze model work around 20 Oct. Submission deadline: 30 Oct 2026 17:00 UTC. The physical arm has not arrived; the repo will be updated with live results when it does.
