#!/usr/bin/env bash
# Multi-task fine-tune with one task held out: POLICY plus TASK demos (scripts/task_demos.py), scored on both
# held_out_tasks suites and the six stacking suites against the policy's own results. A gain on the task that was NOT
# trained is task transfer; the trained task's gain alone is not.
#
#   TASK=basket bash scripts/task_round.sh     # trains on basket, ring held out
#
# Inputs (env, with round 3's defaults): POLICY, TWIN (base demos), RECOVERIES (DAgger recoveries, x UPWEIGHT),
# REF_STACK and REF_TASKS (POLICY's bench files). Demos and the mix are reused when already built.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
set -a; . ./.env; set +a
echo 1000 > /proc/self/oom_score_adj  # on a shared host, this job is the one the OOM killer takes
TASK=${TASK:-basket} EPISODES=${EPISODES:-400} SHARDS=${SHARDS:-8} REPEAT=${REPEAT:-2} UPWEIGHT=${UPWEIGHT:-8}
STEPS=${STEPS:-4000} LEARNING_RATE=${LEARNING_RATE:-1e-5}
POLICY=${POLICY:-$HOME/checkpoints/gr00t-preflight-dagger-r3/checkpoints/004000/pretrained_model}
TWIN=${TWIN:-$HOME/data/preflight_demos_v6_v30} RECOVERIES=${RECOVERIES:-$HOME/data/dagger_recoveries_r3s}
REF_STACK=${REF_STACK:-$HOME/nebius-physical-ai/logs/dagger/r3/bench.json}
REF_TASKS=${REF_TASKS:-$HOME/npai-tasks/logs/held-out-tasks/bench.json}
LR=~/lerobot-n17/.venv/bin VLM=$HOME/checkpoints/hqfang-groot-so100_101/vlm_processor
RAW=$HOME/data/${TASK}_demos_raw MIX=$HOME/data/multitask_${TASK}_mix
OUT=$HOME/checkpoints/gr00t-preflight-multitask-$TASK LOGS=$PWD/logs/multitask-$TASK
STACK_SUITES=(clean_many seen_vocabulary held_out_objects clean_top_down_many real_1_pomade_tin_many in_path_pomade_tin_many)
mkdir -p "$LOGS"
say() { echo "[$(date -u +%T)] task round $TASK: $*"; }

if [ -f "$MIX/meta/info.json" ]; then say "mix already built: $MIX"; else
  say "recording $EPISODES $TASK demos in $SHARDS shards"
  pids=()
  for k in $(seq 0 $((SHARDS - 1))); do
    rm -rf "$RAW/part_$k"
    OMP_NUM_THREADS=1 nice -n 10 uv run --quiet python scripts/task_demos.py "$RAW/part_$k" --task "$TASK" \
      --episodes $((EPISODES / SHARDS)) --shard "$k" > "$LOGS/demos_$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { say "a demo shard failed; see $LOGS/demos_*.log"; exit 1; }; done
  for k in $(seq 0 $((SHARDS - 1))); do
    $LR/python scripts/episode_stats.py "$RAW/part_$k" > /dev/null || exit 1
    $LR/python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id "local/${TASK}_$k" --root "$RAW/part_$k" \
      --push-to-hub false > "$LOGS/convert_$k.log" 2>&1 || { say "convert $k failed"; exit 1; }
  done
  rm -rf "$MIX"
  $LR/python - "$TWIN" "$RECOVERIES" "$RAW" "$MIX" "$TASK" "$SHARDS" "$UPWEIGHT" "$REPEAT" <<'PY' || exit 1
import json, sys
from pathlib import Path
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
twin, recoveries, raw, out, task = sys.argv[1:6]
shards, upweight, repeat = map(int, sys.argv[6:9])
features = json.loads(Path(twin, "meta/info.json").read_text())["features"]
parts = []
for k in range(shards):  # the converter adds per-feature keys the merge won't accept: adopt the base set's schema
    info_path = Path(raw, f"part_{k}", "meta/info.json")
    info_path.write_text(json.dumps({**json.loads(info_path.read_text()), "features": features}, indent=4))
    parts.append(LeRobotDataset(f"local/{task}_{k}", root=f"{raw}/part_{k}"))
base, rec = LeRobotDataset("local/base", root=twin), LeRobotDataset("local/rec", root=recoveries)
# the policy's own training mix beside the new task's demos, so stacking is rehearsed while the task is learnt
merged = merge_datasets([base, *[rec] * upweight, *parts * repeat], f"local/multitask_{task}", output_dir=out,
                        concatenate_videos=False)
print(f"mix: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")
PY
fi

autoresearch-wait --label "task-round-$TASK" --gpu-free-gb 70 --ram-free-gb 20 --stable 120 --timeout 43200 || exit 1
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
# the final save writes 12 GB of weights plus up to 36 GB of optimizer state before it is pruned
[ "$free_gb" -ge 50 ] || { say "only ${free_gb} GB free; the final save needs about 50"; exit 1; }
say "fine-tuning $STEPS steps at lr $LEARNING_RATE"
rm -rf "$OUT"; ln -sfn "$VLM" "$POLICY/vlm_processor"
(cd "$POLICY" && WANDB_PROJECT=preflight WANDB_RUN_GROUP=multitask $LR/lerobot-train \
  --dataset.repo_id="local/multitask_$TASK" --dataset.root="$MIX" \
  --policy.path="$POLICY" --policy.push_to_hub=false --policy.max_steps="$STEPS" --policy.optimizer_lr="$LEARNING_RATE" \
  --rename_map='{"observation.images.front": "observation.images.camera_0", "observation.images.wrist": "observation.images.camera_1"}' \
  --steps="$STEPS" --batch_size=8 --save_freq="$STEPS" --log_freq=20 --num_workers=8 --env_eval_freq=0 \
  --output_dir="$OUT" --job_name="multitask-$TASK" \
  --job.tags="[\"multitask\",\"gr00t-n1.7\",\"lerobot\",\"task:$TASK\",\"sim-only\"]" \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=preflight \
  --wandb.notes="$POLICY fine-tuned $STEPS steps (lr $LEARNING_RATE) on its own training mix plus $EPISODES $TASK demos (x$REPEAT). The other held-out task is not trained: a gain there is task transfer." \
  > "$LOGS/train.log" 2>&1) || { say "training failed; see $LOGS/train.log"; exit 1; }
find "$OUT" -name training_state -type d -exec rm -rf {} + 2>/dev/null

NEW=$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model; ln -sfn "$VLM" "$NEW/vlm_processor"
$LR/python scripts/lerobot_policy_server.py "$NEW" --port 5571 > "$LOGS/serve.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q serving "$LOGS/serve.log"; do kill -0 $SERVER 2>/dev/null || { say "server died"; exit 1; }; sleep 5; done
say "held_out_tasks"
uv run --quiet python scripts/benchmark.py run "multitask-$TASK-tasks" --port 5571 --group held_out_tasks \
  --out "$LOGS/bench_tasks.json" --wandb --wandb-group multitask || { say "tasks benchmark failed"; exit 1; }
uv run --quiet python scripts/benchmark.py compare "$REF_TASKS" "$LOGS/bench_tasks.json"
say "six stacking suites"
only=(); for s in "${STACK_SUITES[@]}"; do only+=(--only "$s"); done
uv run --quiet python scripts/benchmark.py run "multitask-$TASK" --port 5571 "${only[@]}" \
  --out "$LOGS/bench_stack.json" --wandb --wandb-group multitask || { say "stacking benchmark failed"; exit 1; }
uv run --quiet python scripts/benchmark.py compare "$REF_STACK" "$LOGS/bench_stack.json"
say "TASK_ROUND_DONE: $LOGS/bench_tasks.json $LOGS/bench_stack.json"
