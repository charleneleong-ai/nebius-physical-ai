# Raw notes: SO-101 generalisation delta sweep (2026-10-11)

Queries run via the HF MCP (`hf_fs search`):
- papers: "reinforcement learning fine-tuning flow matching VLA", "residual RL vision-language-action", "test-time scaling verifier vision-language-action", "sim-and-real co-training robot manipulation", "SO-101 benchmark generalization", "diffusion steering noise space RL", "GR00T N1.7", "low-cost robot arm SO-100 VLA generalization unseen objects", "RLinf", "DAgger VLA interactive imitation recovery", "co-training data mixture simulation real VLA", "grasp release placement failure VLA", "policy decorator residual RL diffusion"
- models: "so101", "gr00t" (sorted by lastModified)
- datasets: "so101" (sorted by lastModified). Many small per-user teleop sets; nothing large or novel-object-focused surfaced.

WebSearch: LeRobot v0.6 changelog; Isaac GR00T N1.7 / SO-101; SO-101 generalisation benchmark.

## Papers (read at abstract plus results level)
- 2610.10437 SQAM (KAIST/RLWRLD, 10-07). Scalar adjoint, no VJP; OGBench 75 vs 68; real bimanual VLA beats SFT 3/3; residual FT struggled. Code: github.com/yonghdong/sqam
- 2609.34599 Low-rank VLA RL (SNU, 09-28). π0.5 and GR00T N1.5/N1.6. RL updates are low-rank and concentrated in action-expert Timestep Modules; LoRA on them beats standard LoRA; shift-vector probe AUC up to 99.6%.
- 2610.04933 DiVeR (MSR, 10-04). Verifier on action-expert repr z_t; weights w=exp(beta * zscore(var over K candidates)); weighted BCE on trajectory outcome. LIBERO-Long π0 baseline 83.5, π0.5 93.5; π0 N=32 +5.8. RoboCasa N=16 +5.9 / +11.6. Real FR3 π0.5: 58.3 base, 65.6 uniform, 71.9 DiVeR. KDPE / MG-Select often flat or negative.
- 2606.08015 Q-VGM (Jul). Critic-gradient keep-best improved chunks become residual velocity targets; stepwise IQL critic on frozen-backbone latents. π0.5 LIBERO 79.0 to 92.5.
- 2606.18953 Object-centric residual RL (MSRA/KAIST, Jun). Pose-based residual trained in sim with noise and dropout; real FR3 42 to 76 zero-shot; rollouts retrain the base VLA.
- 2610.01178 Recova (NVIDIA, 10-01). Agent-written recovery skills in a digital twin plus DAgger. LIBERO-Pro 78.8 vs 71.7; MolmoSpaces 64.9 vs 38.0; real DAgger 23.8 to 77.5, plus recovery 87.5; intervention 87.5% to 0% over 4 rounds.
- 2609.24118 CARE (DUT, 09-22). Stage-conditioned failure-state distributions, then synthetic corrective demos; FSR-Bench; +14.5 sim / +15.9 real.
- 2608.23478 Indi (POSTECH, Aug). GR00T-N1.7 Bridge 64.3 to 84.7, RoboCasa 64.1 to 70.3.
- 2609.37334 Self-compensating VLA / RoboStress (POSTECH/MBZUAI). Commanded-vs-executed residual; >30 pp real on worn arms.
- 2602.12628 RLinf-Co (Feb). Sim RL plus real SFT anchor; +24 OpenVLA / +20 π0.5 real.
- 2610.11248 SimVLA (SNU, 10-08). Sim-only plus SimDeploy rollouts beat 50 real demos (mobile manipulation).
- 2606.08881 SO-101 VLA benchmark (Jun). Real; π0.5 / SmolVLA / Wall-X / ACT; failure taxonomy.
- 2609.27449 X2Real (Isaac Lab-Arena), 0.84 sim-real correlation. 2607.04434 RoboDojo. SO-101 coverage not confirmed.
- Skimmed, not used: 2609.27033 WTF (image-gen flow maps), 2606.08602 (SVGD flow RL), 2609.37089 Real2Gym (agentic Real2Sim, MuJoCo), 2610.10534 RoboPrompt, 2607.26055 πR².

## Hub (last ~72 h, hackathon cluster)
- maximilianofir/so101-vials-gr00t17-online-ppo-noise010-u16-20261010: Flow-SDE PPO, decoder only (37.9M), 2,048 decisions, actor LR 1e-7, clip 0.1, KL 0.02, privileged critic, potential shaping. Paired 40 scenes: 33/40 vs 33/40, CI [-12.5, +12.5]. The noise025 sibling exists; its eval was not read.
- liorbenhorin-nv/so101_vials_dr_procedural_layout_aware_250ep_20261009_GR00T17_20k: 250 sim episodes, 20k steps, 4x RTX PRO 6000, batch 64, LR 1e-4; card says not evaluated.
- maximilianofir/so101_manual_perception_sim75_real20_GR00T17_20k_20261009: 75 sim + 20 real (10 recovery); sim validation in progress, no eval.
- sreetz-nv/so101_contact_perception_dr500_h32_b32_50k_20261009 (not opened); liorbenhorin-nv/so101_sim_manual_perception_newton_20261008_GR00T17_20k (not opened).
- ducido/GR00T-N1.7-SO101-banana-all49-PbRL, ducido/GR00T-N1.7-SO101-11tasks, twanghcmut/gr00t-n1d7-so101-left-arm-multitask (not opened).

## Releases
- LeRobot v0.6.0 (07-06), v0.6.1 (08-03): GR00T N1.7 replaces N1.5; RL refactor (`sac` to `gaussian_actor`); `lerobot.rewards` (HIL-SERL classifier, SARM, Robometer, TOPReward). https://github.com/huggingface/lerobot/releases/tag/v0.6.0
- Isaac-GR00T: N1.7 current, no N1.8 found. https://github.com/NVIDIA/Isaac-GR00T

## Gaps
- No full-text read of most papers, so numbers come from abstracts and results tables.
- No SO-101 / GR00T N1.7 result found for verifiers, failure synthesis or timestep-LoRA RL.
- No search of X/Discord hackathon channels (not primary sources).
