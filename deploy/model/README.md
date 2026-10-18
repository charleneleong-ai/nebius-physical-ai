---
license: other
license_name: nvidia-license
license_link: LICENSE
base_model: hqfang/groot-so100_101
library_name: lerobot
pipeline_tag: robotics
tags:
- robotics
- lerobot
- groot
- so101
- dagger
- simulation
- physical-ai
language:
- en
model-index:
- name: preflight-gr00t-so101-dagger-r3
  results:
  - task:
      type: robotics
      name: Pick and place on the SO-101 (MuJoCo twin)
    dataset:
      type: preflight-twin-benchmark
      name: Preflight twin benchmark, 96 paired rehearsals per suite, seed 42
    metrics:
      - type: success_rate
        name: pooled, six gate suites (483 of 576)
        value: 0.839
      - type: success_rate
        name: held_out_objects (of 96)
        value: 0.594
      - type: success_rate
        name: seen_vocabulary (of 96)
        value: 0.812
      - type: success_rate
        name: clean_many (of 96)
        value: 0.917
      - type: success_rate
        name: clean_top_down_many (of 96)
        value: 0.885
      - type: success_rate
        name: real_1_pomade_tin_many (of 96)
        value: 0.906
      - type: success_rate
        name: in_path_pomade_tin_many (of 96)
        value: 0.917
      - type: success_rate
        name: held_out_task_basket (of 96)
        value: 0.062
      - type: success_rate
        name: held_out_task_ring (of 96)
        value: 0.000
    source:
      name: W&B report
      url: https://wandb.ai/chaleong/preflight/reports/Preflight:-SO-101-generalisation-experiments--VmlldzoxODA5MDQ3Mg==
---

# Preflight GR00T N1.7 SO-101 — DAgger round 3

The policy behind [Preflight](https://huggingface.co/spaces/chaleong/preflight) and the reference that
[Autopilot](https://huggingface.co/spaces/chaleong/autopilot) improves: GR00T N1.7 for the SO-101 arm, fine-tuned
in a MuJoCo twin to pick a block and place it on another, then improved by three rounds of DAgger (the scripted
expert records recoveries from the policy's own failure states). Built for the Nebius × NVIDIA Physical AI hackathon.
Every experiment and verdict: [Preflight Experiments](https://huggingface.co/spaces/chaleong/preflight-experiments).

**Simulation only.** It has not run on a physical arm.

## Results

Each suite is 96 rehearsals in the randomised twin (12 scenes × 8 perturbations, seed 42), paired scene by scene
across checkpoints.

| suite | v5 (scripted demos) | **this checkpoint** |
|---|---|---|
| held_out_objects (colours and sizes never seen) | 52 | **57** |
| seen_vocabulary | 75 | **78** |
| clean_many | 81 | **88** |
| clean_top_down_many (another camera) | 83 | **85** |
| real_1_pomade_tin_many (a real object as distractor) | 74 | **87** |
| in_path_pomade_tin_many (a real object in the arm's path) | 73 | **88** |
| **pooled** | 438 / 576 | **483 / 576** |
| held_out_task_basket (a task never trained) | – | 6 / 96 |
| held_out_task_ring (a task never trained) | – | 0 / 96 |
| so101-benchmark (external) | 0 / 50 | 0–1 / 50 |

Promoted by a gate fixed before round 1 was scored: pooled paired gain (McNemar) over the served policy and no suite
significantly worse. The new-object gain is one suite of six from one seed; new tasks fail almost every time.

## Training

- **Base:** [hqfang/groot-so100_101](https://huggingface.co/hqfang/groot-so100_101), itself a fine-tune of
  [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B).
- **Data:** 800 scripted twin demos (v5) plus three DAgger rounds of expert recoveries, upweighted ×8; 4,000 steps,
  batch 8, lr 1e-5 (`train_config.json`). Instruction: `Pick blue cube and place on red cube`.
- **Code:** [charleneleong-ai/nebius-physical-ai](https://github.com/charleneleong-ai/nebius-physical-ai)
  (`scripts/dagger_round.sh`, `scripts/benchmark.py`).

## Use

LeRobot 0.6.1 GR00T N1.7 policy; `vlm_processor/` is the base checkpoint's processor, needed to load it.

```bash
git clone https://github.com/charleneleong-ai/nebius-physical-ai && cd nebius-physical-ai
hf download chaleong/preflight-gr00t-so101-dagger-r3 --local-dir checkpoints/dagger-r3
python scripts/lerobot_policy_server.py checkpoints/dagger-r3 --port 5559     # in the LeRobot environment
uv run python scripts/benchmark.py run dagger-r3 --port 5559 --out bench.json
```

## Licence

A derivative of NVIDIA GR00T N1.7, distributed under the [NVIDIA License](LICENSE) that comes with it: **research
and evaluation use only** (non-commercial). The code in the GitHub repository is Apache-2.0; these weights are not.
