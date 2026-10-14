#!/usr/bin/env bash
# FLUX 3 Action with current-frame conditioning and absolute joint targets: tests whether the fair LoRA's closed-loop
# failure (50/576 against v5's 438, though open-loop it beat holding the pose 3 to 1) comes from the checkpoint's 8-step
# self-history and integrated command deltas. Same 800 v4c demos and LoRA recipe as the fair run (flux_fair_chain.sh),
# a quarter of its budget (24k samples in 3k updates of 8), then held_out_objects and clean_many, compared with the fair
# LoRA's result. A clear win at a quarter of the budget implicates the conditioning; a loss leaves the hypothesis open.
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl PATH="$HOME/.local/bin:$PATH"
set -a; . "$HOME/nebius-physical-ai/.env"; set +a
echo 1000 > /proc/self/oom_score_adj
say() { echo "[$(date -u +%T)] flux-frame: $*"; }
PY=$HOME/lerobot-flux/.venv/bin/python
NAME=${NAME:-flux3-lora-frame-abs} STEPS=${STEPS:-6000}
DATA=$HOME/data/preflight_demos_v4c_flux OUT=$HOME/checkpoints/$NAME LOGS=$PWD/logs/$NAME
REF=$HOME/nebius-physical-ai/logs/bench_flux3_lora_v4c.json
mkdir -p "$LOGS"
[ -f "$DATA/meta/info.json" ] || $PY scripts/flux_dataset.py "$HOME/data/preflight_demos_v4c_v30" "$DATA" || { say "bridge failed"; exit 1; }
autoresearch-wait --label flux-frame --gpu-free-gb 60 --ram-free-gb 30 --stable 120 || exit 1
say "LoRA: batch 4 x 2 accumulation x $STEPS microsteps = $((4 * STEPS)) samples, $((STEPS / 2)) updates"
rm -rf "$OUT"
WANDB_PROJECT=preflight WANDB_RUN_GROUP=flux3 $PY scripts/flux_train.py --config_path="$HOME/npai-flux/lora.json" \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --policy.conditioning=frame --policy.condition_on_past_actions=false --policy.action_representation=absolute \
  --policy.delta_absolute_dims=[] --policy.n_obs_steps=1 --policy.history_snapshots=1 \
  --dataset.repo_id=local/preflight_demos_v4c_flux --dataset.root="$DATA" \
  --rename_map='{"observation.images.front": "observation.images.scene"}' \
  --batch_size=4 --accelerator.gradient_accumulation.steps=2 --steps="$STEPS" \
  --dataset.eval_split=0.0 --eval_steps=0 --save_freq=$((STEPS / 2)) --output_dir="$OUT" --job_name="$NAME" \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=preflight \
  --wandb.notes="FLUX 3 Action SO-101 LoRA with current-frame conditioning and absolute joint targets (fresh action heads), same 800 v4c demos and recipe as the fair LoRA (flux3-lora-v4c, 50/576) at a quarter of its budget. Tests whether the fair LoRA's closed-loop failure comes from its 8-step self-history and command deltas." \
  > "$LOGS/train.log" 2>&1 || { say "training failed; see $LOGS/train.log"; exit 1; }
CKPT=$OUT/checkpoints/last/pretrained_model
say "held_out_objects and clean_many"
$PY scripts/flux_eval.py suites "$NAME" --only held_out_objects --only clean_many --checkpoint "$CKPT" \
  --out "$LOGS/bench.json" || { say "suites failed"; exit 1; }
uv run --quiet python scripts/benchmark.py compare "$LOGS/bench.json" "$REF"
uv run --quiet python scripts/benchmark.py failures "$LOGS/bench.json"
say "FLUX_FRAME_DONE"
