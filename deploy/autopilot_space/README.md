---
title: Autopilot
emoji: 🔁
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
thumbnail: https://huggingface.co/spaces/chaleong/autopilot/resolve/main/thumbnail.png
short_description: A closed training loop that widens what a robot can do
---

The self-improving half of [Preflight](https://huggingface.co/spaces/chaleong/preflight): a loop that diagnoses the
served robot policy's failures, proposes one tested training recipe, trains it, and lets a paired gate decide. Built
from `logs/autopilot/cycles.jsonl` by `deploy/autopilot_space/build.py` in
[nebius-physical-ai](https://github.com/charleneleong-ai/nebius-physical-ai).
