---
title: Autopilot
emoji: 🔁
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
thumbnail: https://huggingface.co/spaces/chaleong/autopilot/resolve/main/thumbnail.png
short_description: A closed training loop that widens what a robot can do
models:
- chaleong/preflight-gr00t-so101-dagger-r3
- nvidia/GR00T-N1.7-3B
- hqfang/groot-so100_101
- nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
tags:
- robotics
- agents
- lerobot
- gr00t
- so101
- physical-ai
license: apache-2.0
---

The self-improving half of [Preflight](https://huggingface.co/spaces/chaleong/preflight): a loop that diagnoses the
served robot policy's failures, proposes one tested training recipe, trains it, and lets a paired gate decide. Built
from `logs/autopilot/cycles.jsonl` by `deploy/autopilot_space/build.py` in
[nebius-physical-ai](https://github.com/charleneleong-ai/nebius-physical-ai).
