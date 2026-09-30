# Preflight — design spec

> **Know whether a robot will succeed before it moves.**
> A live webcam scene is rebuilt as a sim twin. A GR00T policy is rehearsed hundreds of times across
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

1. Hands rearrange 3–5 known objects (coloured blocks, a mug) on a marked mat under a fixed webcam.
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
| Policy | GR00T N1.x, fine-tuned on SO-101 sim demos | **Serverless Endpoint** (inference) | GR00T |
| Rehearsal fan-out | MuJoCo + GR00T client, N parallel rollouts | **Serverless Jobs** | sim + policy eval at scale |
| Object/spec lookup | Tavily search (object dimensions, grasp affordances) | Tavily API | Best Use of Tavily bonus |
| UI | Web dashboard: webcam · twin render · rollout grid · verdict | local / Nebius | Design criterion |

Agent runtime: **NVIDIA NeMo Agent Toolkit** wrapping a LangGraph graph
(perceive → twin → rehearse → judge → explain). NeMo is NVIDIA-native, which should land well with
the NVIDIA judges, and LangGraph is what we already know. *Open decision.*

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
| 0 | Sep 30–Oct 4 | Spikes: a Token Factory Nemotron call, a Serverless Job hello-world, Cosmos Reason 2 on the A100 returning scene JSON from a webcam frame, and an SO-101 MuJoCo scene built from that JSON. Pick the agent runtime. |
| 1 | Oct 5–11 | GR00T zero-shot baseline in the twin; scripted demos → LeRobot dataset → fine-tune on the A100; single rollout client. |
| 2 | Oct 12–18 | Serverless Job fan-out (N=200) and results aggregation; trust score and calibration on a held-out scene set (reliability diagram). |
| 3 | Oct 19–25 | Agent graph working end to end, Tavily lookup, dashboard, endpoints deployed on Nebius; first full live run. |
| 4 | Oct 26–30 | Record the video, pitch first, naming Token Factory, AI Cloud, Nemotron, GR00T and Cosmos aloud. Write the README and feedback section, make the repo public under Apache-2.0, and submit by Oct 29. |

## Evaluation

What we claim, and how we'd know it's true:

- **Trust calibration:** ECE of predicted success against actual sim success on held-out scenes.
- **Decision value:** failures avoided against attempts forgone, compared with always-execute and
  random-gate baselines.
- **Latency:** webcam frame to verdict in under 60 s (the fan-out is bounded).

## Risks

| Risk | Mitigation |
|---|---|
| Webcam → twin fidelity | Fixed camera, marked mat and a small set of known objects. Perception reports its confidence, which feeds the trust score. |
| GR00T performs poorly in our twin | Acceptable, because a verifier needs failures to catch. Fine-tune on the A100 only until success is mixed (~30–70%), which makes the verdicts meaningful. |
| Nebius GPU spend | Scale-to-zero endpoints, small N during development, capped Jobs. Claim every credit, including the Builders program. |
| A100 contention with `qmlr` | Schedule fine-tune windows; fall back to a smaller GR00T batch or fewer steps. |
| Cosmos Reason 2 too slow for live use | Cache results per scene. Fall back to Qwen2.5-VL on Token Factory for perception and keep Cosmos for the plausibility check. |

## Open decisions

- Agent runtime: NeMo Agent Toolkit or plain LangGraph.
- GR00T version, and an SO-101 or Franka/Panda twin, whichever has the best pretrained checkpoint.
- Stretch goal: phone-teleop takeover on ASK HUMAN.
