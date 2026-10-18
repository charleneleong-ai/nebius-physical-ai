---
title: Preflight Experiments
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: static
pinned: false
thumbnail: https://huggingface.co/spaces/chaleong/preflight-experiments/resolve/main/thumbnail.png
short_description: Every experiment to make a robot policy generalise, verdicts
models:
- chaleong/preflight-gr00t-so101-dagger-r3
- nvidia/GR00T-N1.7-3B
- hqfang/groot-so100_101
- nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
tags:
- robotics
- lerobot
- gr00t
- so101
- benchmark
- physical-ai
license: apache-2.0
---

Every experiment behind [Preflight](https://huggingface.co/spaces/chaleong/preflight) and
[Autopilot](https://huggingface.co/spaces/chaleong/autopilot): the scoreboard, where gains came from, how the policy
fails, and one verdict per experiment, built from `docs/artifact.html` (`scripts/build_artifact.py`) in
[nebius-physical-ai](https://github.com/charleneleong-ai/nebius-physical-ai).
