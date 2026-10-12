# SO-101 VLA generalisation: SOTA delta sweep

**Status (2026-10-11):** delta sweep covering roughly 2026-08-25 to 2026-10-11, plus a few older items that bear directly on the plan. Sources: HF Papers search, HF Hub model and dataset search, GitHub, vendor blogs. All numbers are vendor-reported (none reproduced by us). Most papers were read at abstract and results-table level only. Raw notes: [`research_notes/so101-generalisation-sota-2026-10-11/notes.md`](../research_notes/so101-generalisation-sota-2026-10-11/notes.md).

**Bottom line.** Nothing new beats HG-DAgger for our setting outright. Three items change how we spend the remaining A100 time: (a) an independent GR00T N1.7 SO-101 online-PPO run in sim measured **0 pp** gain; (b) outcome-trained verifiers now work where density/consensus selection did not (DiVeR); (c) RL gains in flow VLAs, GR00T included, concentrate in the action expert's **timestep modules**.

## What changed

### 1. Post-DAgger RL and interactive fine-tuning of flow/diffusion VLAs

- **Independent GR00T N1.7 SO-101 online PPO: null result.** `maximilianofir/so101-vials-gr00t17-online-ppo-noise010-u16-20261010` ([card](https://huggingface.co/maximilianofir/so101-vials-gr00t17-online-ppo-noise010-u16-20261010)). Flow-SDE PPO (4 stochastic denoising steps, noise 0.10) on the action decoder only (37.9M params), privileged critic, potential-based shaping, 2,048 decisions over 16 batches. Paired eval: **33/40 vs 33/40 base, 0 pp, 95% CI [-12.5, +12.5]**. A sibling run uses noise 0.25. Base: a 250-episode domain-randomised sim GR00T N1.7 policy ([card](https://huggingface.co/liorbenhorin-nv/so101_vials_dr_procedural_layout_aware_250ep_20261009_GR00T17_20k)).
  *Relevance:* this is the closest public analogue to our setup. Short-budget PPO on the decoder does not move a policy that is already at about 80%; our residual PPO needs a larger budget or a narrower target.
- **Low-Rank Structure of VLA RL** ([arXiv 2609.34599](https://arxiv.org/abs/2609.34599), 2026-09-28). RL on π0.5 and GR00T N1.5/N1.6 (LIBERO, ManiSkill, MetaWorld, CALVIN) produces low-rank updates concentrated in the action expert's *Timestep Modules*. Those modules carry a disproportionate share of the RL gain. LoRA aimed at them beats standard LoRA configs. The shift-vector update direction predicts success (ROC-AUC up to 99.6%).
  *Relevance:* if we RL the GR00T head directly, update only the timestep-conditioning modules: a cheaper, lower-variance target.
- **SQAM: Q-learning with Scalar Adjoint Matching** ([arXiv 2610.10437](https://arxiv.org/abs/2610.10437), [code](https://github.com/yonghdong/sqam), 2026-10-07). Off-policy flow fine-tuning that replaces per-step vector-Jacobian products with a closed-form scalar adjoint, plus a value penalty on policy-generated actions. OGBench: 75% vs 68% for the best baseline, +18 to +35 pp on the hardest domains. Offline-to-online VLA fine-tuning on a real bimanual robot beats SFT on 3/3 tasks; residual fine-tuning struggled in the same setting.
  *Relevance:* an off-policy route that reuses every rehearsal rollout we already have (about 576 per round × rounds) instead of fresh on-policy PPO samples.
- **Q-VGM** ([arXiv 2606.08015](https://arxiv.org/abs/2606.08015), Jul 2026; older but missed). Converts critic-gradient-improved action chunks into residual *velocity* targets for the flow head: no likelihoods, no backprop through the denoising chain, no critic at inference. π0.5 few-shot SFT on LIBERO goes from **79.0% to 92.5%** using its own rollouts. Real plug insertion also reported.
  *Relevance:* the same "learn from own rollouts" niche as HG-DAgger, but without needing an expert relabel.
- **Object-centric residual RL for sim-to-real VLA** ([arXiv 2606.18953](https://arxiv.org/abs/2606.18953), MSR Asia, Jun 2026; missed). Frozen VLA plus a residual policy trained in sim on object poses (with pose noise and dropout), transferred zero-shot. Real FR3, 5 tasks: **42% to 76%**. Improved rollouts are then used to retrain the base VLA (self-improvement, no new teleop).
  *Relevance:* validates our Policy-Decorator-style plan, and shows the residual pays off mainly as a *data generator* distilled back into the VLA.
- **Recova** ([arXiv 2610.01178](https://arxiv.org/abs/2610.01178), NVIDIA, 2026-10-01). A coding agent explores failures in a digital twin, writes programmatic recovery skills, and routes rollouts plus human fixes into DAgger. DAgger lifts real success from 23.8% to 77.5%; adding recovery skills reaches 87.5%. Human intervention falls from 87.5% to 0% over 4 rounds.
  *Relevance:* matches our shrinking-recovery curve (85 to 72 to 59). Scripted recovery primitives can replace human relabels once DAgger yield drops.
- **CARE** ([arXiv 2609.24118](https://arxiv.org/abs/2609.24118), [code](https://github.com/xiaojunlan/care), 2026-09-22). Fits *stage-conditioned* distributions of post-failure states from real failed rollouts, then synthesises representative failure states and corrective demos. Reports +14.5 pp in sim and +15.9 pp real across several VLA backbones.
  *Relevance:* a direct recipe for our "holding / pushed / dropped" modes. Sample states from the empirical failure distribution and let the scripted twin expert demonstrate the fix.
- **Indi (Act with Intent)** ([arXiv 2608.23478](https://arxiv.org/abs/2608.23478), Aug 2026). Distils future-based behaviour intent into the VLA decoder. **GR00T-N1.7: SimplerEnv-Bridge 64.3% to 84.7%, RoboCasa 64.1% to 70.3%**; real 62.0% to 68.7%.
  *Relevance:* one of the few recent results on our exact backbone. It is an architecture change, so too heavy for this deadline.

### 2. Test-time scaling and verifiers

- **DiVeR** ([arXiv 2610.04933](https://arxiv.org/abs/2610.04933), Microsoft, 2026-10-04). A lightweight verifier on the action-expert representation, trained with BCE on trajectory-level success labels and reweighted towards *decision-critical* states (high variance across K sampled candidates). Needs only outcome-labelled rollouts from the frozen policy. LIBERO-Long π0 N=32: +5.8 pp over N=1, +2.8 over the best baseline. RoboCasa N=16: +5.9 (π0) and +11.6 (π0.5). Real FR3 π0.5: **58.3% to 71.9%** (uniform-weight verifier 65.6%). Density/consensus selectors (KDPE, MG-Select) were near zero or negative at several budgets.
  *Relevance:* explains our null best-of-8 medoid result (consensus selection is not outcome-grounded), and we already hold the labelled rollouts to train one.

### 3. Sim-to-real and co-training

- **Sim75 + real20 GR00T N1.7 SO-101** ([card](https://huggingface.co/maximilianofir/so101_manual_perception_sim75_real20_GR00T17_20k_20261009)). 75 sim plus 20 real episodes (10 manual, 10 recovery), about 21% real. **Not yet evaluated.** It is a precedent for our 23% mix, not evidence for it.
- **RLinf-Co** ([arXiv 2602.12628](https://arxiv.org/abs/2602.12628), Feb 2026; older). SFT warm-start on sim+real, then sim RL with a real-data SFT loss as an anchor against forgetting. +24 pp (OpenVLA) and +20 pp (π0.5) real over real-only SFT, with better unseen-variation generalisation.
  *Relevance:* if residual or direct RL runs, keep the real-demo SFT term in the loss.
- **SimVLA** ([arXiv 2610.11248](https://arxiv.org/abs/2610.11248), 2026-10-08). Sim-only pretraining plus rollout data (SimDeploy) beats 50 in-domain real demos on zero-shot real mobile manipulation. Supports rollout-derived data over demos alone; mobile manipulation, so transfer to our arm is uncertain.
- **Self-compensating VLA / RoboStress** ([arXiv 2609.37334](https://arxiv.org/abs/2609.37334), 2026-09-29). Online adaptation from the commanded-vs-executed residual (no rewards). >30 pp real on two worn arms, gains extend to unseen objects.
  *Relevance:* SO-101 servos have backlash, so this is the cheapest real-transfer lever if we ever deploy. Not needed for the sim scoreboard.

### 4. GR00T / LeRobot releases, SO-101 checkpoints and datasets

- **LeRobot v0.6.0 (2026-07-06) / v0.6.1 (2026-08-03)** ([release](https://github.com/huggingface/lerobot/releases/tag/v0.6.0), [blog](https://huggingface.co/blog/lerobot-release-v060)). GR00T N1.7 replaces N1.5 (Cosmos-Reason2-2B VLM, flow-matching head, parity-tested against Isaac-GR00T). The RL stack was rebuilt (`sac` became `gaussian_actor`). New `lerobot.rewards` API (HIL-SERL classifier, SARM, Robometer, TOPReward). We are already on 0.6.1, so nothing new since.
- **Isaac-GR00T** ([repo](https://github.com/NVIDIA/Isaac-GR00T), [N1.7 blog](https://huggingface.co/blog/nvidia/gr00t-n1-7)). No N1.8 found. N1.7 remains current.
- **Hub activity in the last 72 h** is dense with SO-101 GR00T N1.7 work tied to the hackathon: DR sim policies, sim+real mixes, online PPO, PbRL (`ducido/GR00T-N1.7-SO101-banana-all49-PbRL`), and an 11-task multitask model (`ducido/GR00T-N1.7-SO101-11tasks`). Most cards report **no evaluation**. None is a drop-in better base than `hqfang/groot-so100_101` on evidence.

### 5. Generalisation benchmarks

- **SO-101 VLA benchmark with failure/recovery analysis** ([arXiv 2606.08881](https://arxiv.org/abs/2606.08881), Jun 2026). Real SO-101, 4 tasks, π0.5 / SmolVLA / Wall-X / ACT, 100 demos per task, 20 trials each, failure taxonomy plus recovery-aware metrics. Varies pose and layout only, not new objects.
  *Relevance:* borrow the failure taxonomy so our holding/pushed/dropped labels map onto a published scheme.
- **X2Real** ([arXiv 2609.27449](https://arxiv.org/abs/2609.27449), Isaac Lab-Arena) reports 0.84 sim-real correlation from calibrated sim. **RoboDojo** ([arXiv 2607.04434](https://arxiv.org/abs/2607.04434)) is a 42-task sim plus 18-task real benchmark. Neither is confirmed to include SO-101. The external so101-benchmark remains the only SO-101-specific sim check we have.

## Ranked candidate next experiments

1. **Outcome-trained verifier for best-of-N (DiVeR-style).** Train an MLP on GR00T action-expert features of executed chunks, with trajectory success labels and candidate-variance weights, using rehearsal rollouts we already have (rounds 1-3, several thousand episodes). Hold out scene seeds from the verifier's training set; otherwise the verifier memorises the eval scenes.
   *Expected:* +3 to +10 pp pooled (paper range), most at the release and approach states.
   *Cost:* about 1 day to extract features and train; eval at N=8-16 costs roughly 8-16× action-head compute (VLM shared via KV cache), within a day on one A100.
   *Targets:* holding and pushed, where candidate disagreement is highest.
2. **Stage-conditioned failure-state synthesis (CARE / Recova-style) to replace shrinking DAgger yield.** Fit per-stage distributions of post-failure states (gripper closed at target, object displaced, object dropped) from round-3 failures, reset the twin into sampled states, and record scripted-expert corrections. Add these as the ×8-upweighted recovery set in round 4.
   *Expected:* restores recovery volume (59 per round is the current bottleneck). Gain on the held-out suites is the test.
   *Cost:* 2-3 days, mostly sim, plus one GR00T fine-tune (about the same as a DAgger round).
   *Targets:* holding, pushed, dropped.
3. **Residual PPO as a data generator, narrowed.** Keep the Policy Decorator run, but (a) restrict the residual's authority to the gripper command plus the final placement phase, and (b) distil successful residual rollouts back into GR00T (object-centric residual RL recipe) rather than deploying the residual. If any direct VLA RL is tried, update only the Timestep Modules via LoRA, and keep an SFT anchor on demos plus real data (RLinf-Co).
   *Expected:* uncertain. The public GR00T SO-101 PPO run is 0 pp. Budget-gate it: abandon if no gain on the dev set after the first 2k decisions at the narrowed scope.
   *Cost:* already running; distillation adds one fine-tune.
   *Targets:* holding (release timing) and dropped.
4. **Off-policy Q-guided flow fine-tuning on existing rollouts (Q-VGM / SQAM).** IQL-style critic on frozen-backbone latents from all logged rehearsals, converted into velocity targets for the GR00T action head. No new environment interaction is needed for the first pass.
   *Expected:* the paper reports +13.5 pp on LIBERO from a weaker start. On a policy at 80%+, expect less.
   *Cost:* 4-6 days of implementation (new code path in the LeRobot GR00T head). Only worth it if #1 shows the critic signal is learnable (verifier AUC well above 0.5 on held-out scenes).
   *Targets:* all modes, especially pushed and holding.
5. **Run the queued co-training (23% real ArmnetBench), but measure it as its own arm.** Keep everything else frozen against the round-2 promoted checkpoint, so the real-data effect on held_out_objects is attributable. Also add the self-compensation adapter only if real deployment happens.
   *Expected:* the held_out_objects suite is the one to watch. No public SO-101 GR00T result yet shows the effect of a similar mix.
   *Cost:* one fine-tune plus one 576-rehearsal eval.
   *Targets:* novel-object perception, not the gripper failures.

## What this changes in the plan

**Changes**
- Replace "medoid consensus best-of-N" with an **outcome-trained verifier** before writing off test-time scaling. Consensus selection is the weak baseline in DiVeR's own tables.
- **Gate the residual PPO** on an early dev-set check and narrow its scope to release and placement. The closest public analogue measured 0 pp at a similar budget.
- Plan round 4 of HG-DAgger around **synthesised failure states**, not just new rollouts, because recovery yield is falling 85 to 72 to 59.
- If we do direct RL on GR00T, use **Timestep-Module LoRA plus an SFT anchor**, not full-decoder updates.

**Doesn't change**
- GR00T N1.7 via LeRobot 0.6.1 with `hqfang/groot-so100_101` stays the base. No newer release, and no SO-101 checkpoint with published evaluation beats it.
- HG-DAgger with ×8 recovery upweighting stays the main lever. Nothing in the sweep shows a cheaper intervention with larger reported gains on held-out objects.
- The FLUX 3 Action LoRA stays closed (closed-loop collapse). Nothing new addresses open-loop/closed-loop mismatch for that model.
- Paired McNemar on the six suites × 96 remains the promotion gate. Benchmark papers add taxonomy, not a better statistic.
- Co-training with the 400 real demos goes ahead as queued.

**Not established:** none of the 2026-09/10 papers report SO-101 or GR00T N1.7 results for verifiers, CARE-style synthesis, or Timestep-Module RL. Transfer to our arm and backbone is the hypothesis each experiment tests.
