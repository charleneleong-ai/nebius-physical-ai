#!/usr/bin/env bash
# Record sim demos, build GR00T stats, fine-tune, logging to W&B project "preflight".
# Runs on the A100 from the repo root:  scripts/finetune.sh <name> [episodes] [max_steps]
# Saves a single checkpoint at the end: each is ~22 GB and the A100 disk is tight.
# Needs ~/Isaac-GR00T on branch n1d6 (>= 9b37aa1) with its .venv, and the base checkpoint below.
set -euo pipefail
NAME=${1:?usage: scripts/finetune.sh <name> [episodes] [max_steps]}
EPISODES=${2:-100}
MAX_STEPS=${3:-2000}
BASE=${GR00T_BASE:-$HOME/checkpoints/gr00t-n1.6-so101-multitask}
GR00T=${GR00T_DIR:-$HOME/Isaac-GR00T}
DATA=$HOME/data/preflight_demos_$NAME
OUT=$HOME/checkpoints/gr00t-preflight-$NAME

rm -rf "$DATA"
MUJOCO_GL=egl uv run npai demos "$DATA" --episodes "$EPISODES" --seed 1

cd "$GR00T"
.venv/bin/python gr00t/data/stats.py --dataset-path "$DATA" --embodiment-tag NEW_EMBODIMENT || true  # rel stats need the config below
.venv/bin/python - "$DATA" <<'PY'
import sys
sys.path.insert(0, "examples/SO100")
import so100_config  # noqa: F401 - registers the NEW_EMBODIMENT modality config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats
generate_rel_stats(sys.argv[1], EmbodimentTag.NEW_EMBODIMENT)
PY

WANDB_RUN_GROUP=$NAME .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" --dataset-path "$DATA" \
  --modality-config-path examples/SO100/so100_config.py --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 1 --output-dir "$OUT" --max-steps "$MAX_STEPS" --save-steps "$MAX_STEPS" --save-total-limit 1 \
  --global-batch-size 32 --dataloader-num-workers 8 --learning-rate 1e-4 \
  --use-wandb --wandb-project "${WANDB_PROJECT:-preflight}"
echo "FINETUNE_DONE $OUT/checkpoint-$MAX_STEPS"
