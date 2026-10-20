# FLUX 3 Action in Preflight: gap analysis (2026-10-18)

Read-only analysis. Nothing was trained or run. Our evidence comes from [#18](https://github.com/charleneleong-ai/nebius-physical-ai/pull/18), [#32](https://github.com/charleneleong-ai/nebius-physical-ai/pull/32), [`scripts/flux_eval.py`](../scripts/flux_eval.py), [`scripts/flux_audit.py`](../scripts/flux_audit.py), [`scripts/flux_dataset.py`](../scripts/flux_dataset.py) and [`docs/experiments/so101/verdicts.yaml`](../docs/experiments/so101/verdicts.yaml).

## 0. What the published results actually cover

All vendor numbers below are BFL-reported and have not been reproduced independently.

- **The RoboLab result belongs to a different checkpoint.** RoboLab-120 is scored with the DROID policy: Franka, three cameras, absolute joint-position commands, 15 Hz, 32-step chunks, fine-tuned at batch 512 to 2048 for 20k steps. It reports 42.92% in the table and 41.6 / 42.2 ± 0.4 in the text, against Cosmos3-Nano 36.8% and GR00T N1.6 7.2%. Separately, the Franka real-robot test went 28/30. Sources: [bfl.ai/models/flux-3-action](https://bfl.ai/models/flux-3-action), [HF blog](https://huggingface.co/blog/black-forest-labs/flux-3-action).
- **No SO-101 success rate is published.** The SO-101 evidence is videos only, from a policy "adapted on about 200 teleoperated episodes covering a handful of related pick-and-place tasks" ([blog](https://huggingface.co/blog/black-forest-labs/flux-3-action)). The camera-robustness claim is shown in clips and never measured.
- **Simulation is outside the base model's training.** The base "was not trained on simulated data" ([blog](https://huggingface.co/blog/black-forest-labs/flux-3-action)). Midtraining used teleop from 14 embodiments plus gaming and egocentric video. Without pretraining, RoboLab success stays below 1%.

So the strong published result does not tell us how the SO-101 checkpoint behaves in a MuJoCo twin.

## 1. Published operating conditions vs ours

| Axis | FLUX SO-101 checkpoint ([card](https://huggingface.co/black-forest-labs/flux-3-action-so101), [LeRobot docs](https://huggingface.co/docs/lerobot/main/en/flux3)) | Ours | Already tested? |
|---|---|---|---|
| History | 8 state steps, 2 visual snapshots, past-command conditioning | Same, but the history is produced by its own actions, starting from rest | Partly: #32 drops it (3 → 21) |
| Actions | Joint **deltas** integrated from the last command, absolute gripper, quantile-normalised (clip 6) | Same; DROID/RoboLab uses absolute joints | Partly: #32, confounded with history |
| Units | Community-dataset calibration | Affine `QuantileBridge` on state and action | Yes: lossless round trip, 98–100% in range |
| Rate / chunk | 30 Hz; predict 42, execute 32 | 30 Hz; 32 (16 tried) | Yes: 16 is not better |
| Cameras | Scene + wrist on a 256×512 canvas, real | Rendered scene + wrist at 240×320, oblique | No |
| Embodiment / domain | Real SO-101 teleop; no sim in pretraining | MuJoCo twin, scripted demos | Imagination L1 is similar zero-shot and LoRA |
| Data | ~200 teleop episodes, few tasks | 800 scripted demos, many objects | Volume matched to GR00T v5 |
| Recipe | LoRA r32 + full heads, EMA 0.999, pick raw vs EMA by rollouts, guidance 3/3 (4/1 best on RoboLab) | Raw `pretrained_model`, guidance 3/3; #32 used ¼ budget with re-initialised heads | No: EMA and guidance untried |

## 2. Ranked hypotheses for the closed-loop collapse

1. **A fixed point at rest from self-history plus deltas (copycat).** Evidence for it:
   - 358 of 526 failures are `no_contact`, meaning the block never moved.
   - The open-loop audit samples `t ≥ start + 8` and feeds demo history. The episode start, an arm at rest with command equal to state, is exactly the state it never scores, and its own docstring says that without history it "starts late".
   - Removing history and deltas lifted held_out from 3 to 21 (p < 0.001).
2. **Delta undershoot that compounds.** Open loop, the chunks "undershoot the magnitude" (#18). Integrated deltas carry that shortfall forward, so the arm creeps. Rollouts cap at 320 ticks (10.7 s), so a slow approach is also recorded as `no_contact`. Scripted deltas may also sit in a narrow band of the checkpoint's teleop q01–q99 range: they are in range but at low resolution.
3. **Under-trained heads in #32.** The heads were re-initialised from scratch and trained on 24k samples, a quarter of the fair run. Failures moved from `no_contact` to `pushed` (48 of 75), so the arm now approaches but grasps imprecisely.
4. **Recipe gaps.** EMA weights were never scored, nor guidance 4/1. The preset holds out the last 20% of episodes, so the run may have trained on 640 demos rather than 800 (to be checked).
5. **Perception and resolution: lowest priority.** Imagination L1 is the same zero-shot and fine-tuned (16.5 vs 15.4), so scene understanding is not the bottleneck. A 256-px half-canvas from an oblique 320-px render could still hurt grasp precision.

## 3. Diagnostic experiments, cheapest first

| # | Cost | Experiment | Confirms / kills |
|---|---|---|---|
| 1 | CPU, minutes | Re-read existing rollout videos and logs ([hqt3v5n5](https://wandb.ai/chaleong/preflight/runs/hqt3v5n5), [koghf3ss](https://wandb.ai/chaleong/preflight/runs/koghf3ss)): ticks to first gripper motion, closest gripper–block distance in `no_contact` runs. Also check `train_config.json` for the 20% hold-out. | An arm stalled near its start pose supports H1. Steady but slow approach supports H2. |
| 2 | CPU | Extend `flux_audit.py bridge`: histogram the per-joint delta z-scores of our demos against the checkpoint's band. | If most \|z\| is below 0.05, H2's resolution problem is real and absolute targets are justified. |
| 3 | GPU inference, about 1 h | Run `openloop` at **t = 0** with rest history, and again at t = 8, 32. | If FLUX predicts roughly hold-pose at t=0 but correct chunks later, H1 is confirmed directly. |
| 4 | GPU inference, about 2–3 h | **Teacher-forced hand-off**: replay demo actions for k ∈ {0, 8, 32, 64} ticks, then let FLUX drive, on held_out_objects. Add a 1.3× delta gain and EMA + guidance 4/1 arms. | Success rising with k means start-from-rest (H1). Gain helping means undershoot (H2). EMA/guidance helping means H4. Flat across all arms kills H1, H2 and H4 and points to H3/H5. |
| 5 | Training, about 5 h each | 2×2 split of #32: {history, frame} × {delta, absolute} at equal budget, then the best cell at full budget with EMA. | Separates the two changes, and tests whether budget closes 21 → 57. |

## 4. A fair shot, or stop

A fair shot needs restored gated access: BFL's repos now return 403 for this account (#32, and the HF MCP here). It also needs experiment 5's best cell at full budget with EMA, and ideally the DROID-style absolute contract, since that is the representation the published RoboLab result used.

**Recommendation: run experiments 1–4 (under a day, no training).** Stop FLUX unless experiment 4 shows a lever worth more than about 20 successes on held_out. Even at that point it would still need to roughly triple, from 21 to round 3's 57, and the 2026-10-30 deadline favours spending GPU time on GR00T.
