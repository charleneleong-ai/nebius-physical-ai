---
title: Preflight
emoji: 🤖
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Know whether the robot will succeed before it moves
thumbnail: https://huggingface.co/spaces/chaleong/preflight/resolve/main/thumbnail.png
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
- physical-ai
---

Preflight rebuilds a tabletop from one photo as a MuJoCo twin, rehearses the robot's GR00T N1.7 policy there, and
answers GO, RETRY or ASK_HUMAN with Nemotron's explanation. Code and results:
https://github.com/charleneleong-ai/nebius-physical-ai
