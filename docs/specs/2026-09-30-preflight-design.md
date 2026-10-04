# Preflight — design spec

> **Know whether a robot will succeed before it moves.**
> A phone pointed at a real workspace streams the scene, which is rebuilt as a sim twin. A GR00T policy is rehearsed hundreds of times across
> randomised variants on Nebius Serverless Jobs. A Nemotron agent turns the rehearsal into a
> calibrated go / retry / ask-a-human verdict it can explain.

Deadline **Fri 30 Oct 2026, 17:00 UTC**. Rules: [`../hackathon-rules.md`](../hackathon-rules.md).

## Problem & audience

Robot policies such as GR00T fail silently and at real cost: a dropped part, a collision, a stalled
line. Operators of small fleets (warehouse picking, lab automation, light assembly) get no signal for
*"will this attempt work in this scene, right now?"* until after it fails. Preflight is a
pre-execution check. It rehearses in a twin of the live scene, estimates the chance of success, and
only acts when trust is high.

This carries over a theme from prior work: the policy is the commoditised part, and the **verifier**
is where the reliability comes from (touchstone, ContinualWAM trust weighting). In the PlugCharger
study, imitation policies succeeded only 0.13–0.17 of the time. Catching failures like those is the
verifier's job.

## Demo loop (what the video shows, ≥1 min live)

1. Hands rearrange 3–5 known objects (coloured blocks, a mug) on a mat with ArUco markers at the
   corners. A phone streams the scene from a browser page (no app install); Continuity Camera is the
   fallback. The markers give camera pose per frame, so the phone can be hand-held.
2. **Perceive.** Cosmos Reason 2 parses the frame into a scene graph (objects, poses, relations)
   and flags physically implausible layouts.
3. **Twin.** The scene graph becomes a MuJoCo SO-101 scene, reusing the nebius/nebius-physical-ai
   assets.
4. **Rehearse.** A Serverless Job runs N≈200 GR00T rollouts in parallel, each with a randomised
   variant (pose noise, friction, mass, lighting). It records success, contacts and near-misses.
5. **Judge.** Trust score = calibrated success probability × twin fidelity (perception confidence,
   rollout variance). Verdict: **GO**, **RETRY with an alternative plan**, or **ASK HUMAN**.
6. **Explain.** The Nemotron agent explains the verdict, for example: "mug handle faces away; 38%
   grasp success; 3 of 5 failure clusters are wrist collisions". It proposes a rearrangement. On
   ASK HUMAN it hands control to phone teleop (stretch goal).
7. A human moves an object in response and the loop re-runs live. That live reaction is the key
   shot of the video.

## Architecture

| Stage | Model / service | Runs on | Track requirement satisfied |
|---|---|---|---|
| Orchestration agent | Nemotron 3 Super (fast calls) / Ultra (explanations) | **Token Factory** | Nemotron + agent runtime |
| Scene perception | Cosmos Reason 2 (container) | **Serverless Endpoint** | Cosmos, real-time inference |
| Policy | GR00T **N1.6** (`nvidia/GR00T-N1.6-3B`, `NEW_EMBODIMENT`, SO-100/101 modality config), fine-tuned on SO-101 sim demos | **Serverless Endpoint** (inference) | GR00T |
| Rehearsal fan-out | MuJoCo + GR00T client, N parallel rollouts | **Serverless Jobs** | sim + policy eval at scale |
| Object/spec lookup | Tavily search (object dimensions, grasp affordances) | Tavily API | Best Use of Tavily bonus |
| Capture + UI | Phone web app: camera stream, verdict card, teleop on ASK HUMAN; desktop dashboard: twin render · rollout grid | local / Nebius | Design criterion |

Agent runtime: **NVIDIA NeMo Agent Toolkit** wrapping a LangGraph graph
(perceive → twin → rehearse → judge → explain). NeMo is NVIDIA-native, which should land well with
the NVIDIA judges, and LangGraph is what we already know. *Open decision.*

## Hardware

**Robonine SO-ARM101 kit** (ordered 2026-09-30, ships unassembled from China): standard Feetech
STS3215 follower + leader arm + follower wrist camera. Standard motors matter — LeRobot's SO-101
driver, NVIDIA's [SO-101 sim-to-real course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html)
and the community checkpoint
[`quangnd58/GR00T-N1.6-3B-so101-multitask`](https://huggingface.co/quangnd58/GR00T-N1.6-3B-so101-multitask)
all assume them. GR00T N1.6 is pinned because the SO-101 examples live on the `n1d6` branch
(main now defaults to N1.7). The leader arm is insurance: if sim-to-real transfer is poor, record
20–50 real demos. Execution after GO runs through LeRobot (`--policy.type=gr00t`).

Hardware is additive: if the arm is late or misbehaves, the submission falls back to sim-only
("key application modules in action"), which the rules allow.

## Compute plan

This hackathon has no GPU or fine-tuning credit. Checked 2026-09-30, the only credits are
$25 + $25 of Token Factory, $25 of Tavily, and LangSmith/Toloka credit. AI Cloud is pay-as-you-go.

- **pi-a100-80gb** handles development, GR00T fine-tuning and Cosmos Reason 2 prototyping.
  - The GPU is shared: a `qmlr` job uses 99% of compute but only ~5 GB of VRAM.
  - Disk is 85% full with 112 GB free, and the GR00T and Cosmos checkpoints need ~40 GB.
  - Coordinate with or pause qmlr during fine-tune windows.
- **Nebius AI Cloud** must host the *demo runtime*, because running on the A100 does not count as
  "runs on Nebius".
  - Endpoints scale to zero between sessions. Jobs run only during rehearsals.
  - Target out-of-pocket budget is **≤ $150** (see risks).

## Reuse

| From | Asset |
|---|---|
| `robotics_world_models/experiments/diffusion_wm/fidelity.py` | calibration, divergence detection, trust scoring |
| `robotics_world_models/experiments/causal_trust_world_model_learning/boundary_cusum.py` | drift detection between twin and live scene |
| `robotics_world_models/experiments/diffusion_wm/domain_rand.py` | randomisation schedule for rehearsal |
| `smart_city_foundation_model` | conformal calibration pattern |
| `nebius/nebius-physical-ai` | SO-101 MuJoCo assets, LeRobot synthetic-data export, MuJoCo physics judge |
| `autoresearch` | sweep tracking for fine-tune runs |

Assets are imported, or copied with a note on their origin. They are never nested.

## Plan

| Week | Dates | Deliverable (exit criterion) |
|---|---|---|
| 0 | Sep 30–Oct 4 | Spikes: a Token Factory Nemotron call, a Serverless Job hello-world, phone web app streaming frames to the backend with ArUco pose, Cosmos Reason 2 on the A100 returning scene JSON from a frame, and an SO-101 MuJoCo scene built from that JSON. Pick the agent runtime. |
| 1 | Oct 5–11 | GR00T N1.6 community SO-101 checkpoint baseline in the twin; scripted demos → LeRobot dataset → fine-tune on the A100; single rollout client. |
| 2 | Oct 12–18 | Serverless Job fan-out (N=200) and results aggregation; trust score and calibration on a held-out scene set (reliability diagram). |
| 3 | Oct 19–25 | Arm assembled + calibrated (leader/follower teleop sanity, LeRobot driver); agent graph working end to end, Tavily lookup, dashboard, endpoints deployed on Nebius; first full live run. |
| 4 | Oct 26–30 | Record the video, pitch first, naming Token Factory, AI Cloud, Nemotron, GR00T and Cosmos aloud. Write the README and feedback section, make the repo public under Apache-2.0, and submit by Oct 29. |

## Related work and positioning

Twin-based policy evaluation already exists; the simulation is not the novelty.

| Work | What it does | Gap Preflight fills |
|---|---|---|
| [SIMPLER](https://simpler-env.github.io/), [PolaRiS](https://arxiv.org/pdf/2512.16881), [Real-to-Sim Eval](https://real2sim-eval.github.io/) | Scanned twins whose sim success rates track real ones across **policies** (r ≈ 0.9) | They rank policies offline, on average. Preflight decides on **one attempt in one scene**. |
| [Real-is-Sim](https://huggingface.co/papers/2504.03597), [TwinRL](https://arxiv.org/pdf/2602.09023) | Digital twin inside training and deployment pipelines | Train/evaluate, not a per-attempt go/no-go |
| [FIPER](https://arxiv.org/abs/2510.09459), [SAFE](https://openreview.net/forum?id=XPyAukgsFf), [VLA-Scope](https://arxiv.org/html/2609.21246) | Conformally calibrated failure alarms **during** execution | They fire after the robot moves. Preflight predicts **before**. |

Claim: a calibrated, pre-execution probability that *this* attempt succeeds, with abstention
(RETRY vs ASK HUMAN) and an agent that acts on it. [RoboMonitor](https://arxiv.org/html/2609.30715v1)
lists conformal abstention as future work. Never claim "accurate simulation" or "first digital
twin". Stretch: FIPER-style drift alarm after GO, so the attempt is covered before and during.

## Trust score

Two separate questions, kept separate on purpose:

- **p: will it work?** Successes over N randomised rehearsals, with a Wilson interval, then
  calibrated against real outcomes (isotonic regression, or conformal as in FIPER).
- **f: can I trust the twin for this scene?** Weakest-link combination of perception confidence
  (markers, detector), scene match (twin render vs phone frame), variant disagreement,
  out-of-distribution distance from validation scenes, and execution drift (`boundary_cusum.py`).

| Verdict | Rule (thresholds tuned on validation data, not hand-picked) |
|---|---|
| GO | lower bound of p ≥ 0.7 and f ≥ 0.6 |
| RETRY | f ≥ 0.6 but p low: Nemotron proposes a rearrangement, re-rehearse |
| ASK HUMAN | f < 0.6: the twin is not trustworthy here, so don't guess |

Week 2 builds p with its interval, perception confidence and variant disagreement; the other
fidelity signals follow once the core loop works.

## Evaluation

What we claim, and how we'd know it's true. Report absolute counts; no ratio metrics on small samples.

- **Real-arm validation (week 3):** 20–30 mat layouts, each with a Preflight prediction and a real
  attempt. Reliability diagram, Brier score, and rank agreement (does higher p mean more successes?).
- **Sim-to-sim stress test (week 1–2, before the arm):** a deliberately mismatched "reality" twin
  (friction, mass, gains, camera pose). Does domain randomisation keep p calibrated there?
- **Decision value (headline number):** failures avoided against attempts wrongly skipped, versus
  always-execute and random-gate baselines.
- **Fidelity check:** low-f scenes should show larger prediction error; if not, drop that signal.
- **Latency:** phone frame to verdict in under 60 s (the fan-out is bounded).
- **Demo moment:** Preflight predicts RETRY, an override fails on the real arm; after the suggested
  rearrangement it predicts GO and the arm succeeds.

## Risks

| Risk | Mitigation |
|---|---|
| Phone → twin fidelity | ArUco-marked mat for camera pose, small set of known objects, stable frame selection (skip blurred frames). Perception reports its confidence, which feeds the trust score. |
| GR00T performs poorly in our twin | Acceptable, because a verifier needs failures to catch. Fine-tune on the A100 only until success is mixed (~30–70%), which makes the verdicts meaningful. |
| Camera viewpoint mismatch (tripod height/tilt, wrist-mount tolerance, field of view) | Randomise top-camera pose (±3 cm, ±5° tilt/yaw) and FOV (±5°) plus wrist-mount offsets in demos and rehearsals; centre the range on the real top camera's pose, measured from the ArUco markers; lower fidelity when the measured pose falls outside the trained range. |
| Nebius GPU spend | Scale-to-zero endpoints, small N during development, capped Jobs. Claim every credit, including the Builders program. |
| A100 contention with `qmlr` | Schedule fine-tune windows; fall back to a smaller GR00T batch or fewer steps. |
| Cosmos Reason 2 too slow for live use | Cache results per scene. Fall back to Qwen2.5-VL on Token Factory for perception and keep Cosmos for the plausibility check. |

## Open decisions

- Agent runtime: NeMo Agent Toolkit or plain LangGraph.
- Stretch goal: simulated Unitree G1 humanoid rehearsed with GR00T + **GEAR-SONIC** whole-body control (NVlabs/GR00T-WholeBodyControl) — covers all four named track models.
- Stretch goal: phone-teleop takeover on ASK HUMAN (same phone page, IMU or touch control).
