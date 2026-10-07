# Preflight

**Know whether a robot will succeed before it moves.** Entry for the [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) — **Physical AI track**.

> Build embodied and edge agents that sense and act in the real world: robotics, IoT, and on-device
> intelligence, powered by Nemotron, GR00T, Cosmos and Sonic models and coordinated through an agent
> runtime. Use Nebius Serverless Jobs to run simulations, generate synthetic data, evaluate robot
> policies, or process sensor data at scale, and Nebius Serverless Endpoints to serve real-time
> inference when needed.

## How Preflight uses Nebius and NVIDIA

A phone photographs the workspace, the scene is rebuilt as a MuJoCo twin of the SO-101 arm, a GR00T policy rehearses
the task dozens of times across randomised variants, and the calibrated success rate becomes a **GO / RETRY / ASK_HUMAN**
verdict that a Nemotron agent explains.

| Stage | Model | Runs on | Code |
|---|---|---|---|
| Verdict explanation | **Nemotron 3 Ultra** (Super / 3.5 Lightning selectable) | **Nebius Token Factory**, called on every verdict | [`NemotronAgent`](src/nebius_physical_ai/pipeline.py) |
| Scene perception | **Cosmos 3** Reasoner, with a Token Factory VLM as an alternative | our GPU (vLLM) or **Nebius Token Factory** | [`CosmosClient`, `TokenFactoryVLClient`](src/nebius_physical_ai/pipeline.py), [`scripts/cosmos_local.sh`](scripts/cosmos_local.sh) |
| Policy | **GR00T N1.7** (v5, fine-tuned from [hqfang/groot-so100_101](https://huggingface.co/hqfang/groot-so100_101)); GR00T N1.6 for v1–v4c | our A100 | [`scripts/lerobot_policy_server.py`](scripts/lerobot_policy_server.py), [`scripts/finetune.py`](scripts/finetune.py) |
| Rehearsal, benchmark, trust score | — | our A100 (MuJoCo, headless) | [`twin/`](src/nebius_physical_ai/twin), [`benchmark.py`](src/nebius_physical_ai/benchmark.py), [`trust/`](src/nebius_physical_ai/trust) |

Every verdict makes a runtime call to Token Factory; without a `NEBIUS_API_KEY` the explanation step fails rather than
falling back to canned text (`--offline-explain` exists for offline tests only). GR00T and Cosmos run on our own GPU,
which the [FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs) allows alongside a Token Factory call. Model
weights are downloaded from Hugging Face and run by us; no NVIDIA model is reached through a third-party inference
provider. Submission status: [`docs/hackathon-rules.md`](docs/hackathon-rules.md#submission-checklist).

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
