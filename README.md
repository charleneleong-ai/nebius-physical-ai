# Preflight

**Know whether a robot will succeed before it moves.** Entry for the [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) — **Physical AI track**.

> Build embodied and edge agents that sense and act in the real world: robotics, IoT, and on-device
> intelligence, powered by Nemotron, GR00T, Cosmos and Sonic models and coordinated through an agent
> runtime. Use Nebius Serverless Jobs to run simulations, generate synthetic data, evaluate robot
> policies, or process sensor data at scale, and Nebius Serverless Endpoints to serve real-time
> inference when needed.

## Track requirements checklist

- [ ] NVIDIA model(s): Nemotron / GR00T / Cosmos / Sonic
- [ ] Agent runtime coordinating the models
- [ ] Nebius **Serverless Jobs** — sim / synthetic data / policy eval / sensor processing at scale
- [ ] Nebius **Serverless Endpoints** — real-time inference
- [ ] Demo video with ≥1 min of hardware operating (or key application modules in action)

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in Nebius / Tavily keys
uv run npai info
uv run pytest
```

Hackathon rules digest: [`docs/hackathon-rules.md`](docs/hackathon-rules.md). Design spec: [`docs/specs/2026-09-30-preflight-design.md`](docs/specs/2026-09-30-preflight-design.md).

## Policy

The served policy is **v5**: [hqfang/groot-so100_101](https://huggingface.co/hqfang/groot-so100_101) (GR00T N1.7,
fine-tuned on 1,209 SO-100/101 datasets) fine-tuned again with LeRobot on 800 scripted demos recorded in the SO-101 twin.
On the paired benchmark ([`scripts/benchmark.py`](scripts/benchmark.py), 96 rehearsals per suite, the same scenes for
every checkpoint) it beats the earlier Isaac-GR00T fine-tunes on every suite, including objects never seen in training
(57% against 25% for v4c). All results so far are in simulation; the arm is not built yet.

v5 is a LeRobot checkpoint, so it is served by [`scripts/lerobot_policy_server.py`](scripts/lerobot_policy_server.py),
which speaks the Isaac-GR00T ZeroMQ protocol: the rehearsal, benchmark and verdict code talk to it exactly as to an
Isaac-GR00T server. It runs in its own LeRobot environment on a GPU host:

```bash
uv venv ~/lerobot-n17/.venv --python 3.12
uv pip install --python ~/lerobot-n17/.venv/bin/python "lerobot[groot,training]==0.6.1"   # keeps wandb below 0.28
CKPT=~/checkpoints/gr00t-preflight-v5/checkpoints/012000/pretrained_model
ln -sfn ~/checkpoints/hqfang-groot-so100_101/vlm_processor "$CKPT/vlm_processor"   # saved checkpoints omit it
~/lerobot-n17/.venv/bin/python scripts/lerobot_policy_server.py "$CKPT" --port 5558
```

Reproducing v5 from the base checkpoint: convert the demos to LeRobot v3.0 ([`scripts/episode_stats.py`](scripts/episode_stats.py)
then `lerobot.scripts.convert_dataset_v21_to_v30`), recompute the checkpoint's joint normalisation from them
([`scripts/restat_checkpoint.py`](scripts/restat_checkpoint.py): the base pools its statistics in degrees, the twin
records LeRobot units), then `lerobot-train --policy.path=<re-statted checkpoint>` with the front and wrist cameras
renamed to `camera_0` and `camera_1`. Isaac-GR00T fine-tunes (v1 to v4c) are served with `scripts/finetune.py serve`.

## Licences

This repository's code is [Apache-2.0](LICENSE). The model weights are not covered by it:

- **v5 and its base** derive from [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) and are under the
  NVIDIA licence that comes with it, which restricts use to research and evaluation. They are not redistributed here.
- **v1 to v4c** derive from GR00T N1.6 via the community checkpoint
  [quangnd58/GR00T-N1.6-3B-so101-multitask](https://huggingface.co/quangnd58/GR00T-N1.6-3B-so101-multitask), under the
  terms of its base model.
- **Cosmos 3** and **Nemotron** are used under their own NVIDIA model licences; Nemotron is called through Nebius Token
  Factory.
