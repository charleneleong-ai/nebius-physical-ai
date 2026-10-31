#!/usr/bin/env bash
# Train the served policy on the pick/place pairs Preflight refused, grouped by pair and weighted by how many
# refusals asked for each. One GPU run covers every outstanding request instead of one run per photo.
#
#   PAIRS="yellow:red:3,blue:yellow:1" bash scripts/pair_round.sh
#
# Each entry is pick:place[:weight]; weight defaults to 1 and only sets that pair's share of EPISODES. The mix is
# the policy's own training set plus the new demos, so stacking is rehearsed while the pairs are learnt -- cycle 1
# showed a task trained without that rehearsal costs the old skill (clean_many 88 -> 76, p=0.004).
#
# This trains on what was refused, so it closes specific refusals; it is not evidence of transfer to other tasks.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
set -a; . ./.env; set +a
echo 1000 > /proc/self/oom_score_adj  # on a shared host, this job is the one the OOM killer takes
PAIRS=${PAIRS:?PAIRS="pick:place[:weight],..." is required}
EPISODES=${EPISODES:-400} SHARDS=${SHARDS:-8} REPEAT=${REPEAT:-2} UPWEIGHT=${UPWEIGHT:-8}
STEPS=${STEPS:-4000} LEARNING_RATE=${LEARNING_RATE:-1e-5}
POLICY=${POLICY:-$HOME/checkpoints/gr00t-preflight-dagger-r3/checkpoints/004000/pretrained_model}
TWIN=${TWIN:-$HOME/data/preflight_demos_v6_v30} RECOVERIES=${RECOVERIES:-$HOME/data/dagger_recoveries_r3s}
REF_STACK=${REF_STACK:-$HOME/nebius-physical-ai/logs/dagger/r3/bench.json}
CAMERA=${CAMERA:-$PWD/configs/front_camera.json}
LR=~/lerobot-n17/.venv/bin VLM=$HOME/checkpoints/hqfang-groot-so100_101/vlm_processor
TAG=$(echo "$PAIRS" | tr ',:' '--')
RAW=$HOME/data/pairs_${TAG}_raw MIX=$HOME/data/pairs_${TAG}_mix
OUT=$HOME/checkpoints/gr00t-preflight-pairs-$TAG LOGS=$PWD/logs/pairs-$TAG
STACK_SUITES=(clean_many seen_vocabulary held_out_objects clean_top_down_many real_1_pomade_tin_many in_path_pomade_tin_many)
mkdir -p "$LOGS"
say() { echo "[$(date -u +%T)] pair round $TAG: $*"; }
die() { say "$*"; exit 1; }

# the datasets the mix is built from, before anything spends GPU time on them
[ -f "$TWIN/meta/info.json" ] || die "no twin dataset at $TWIN"
[ -f "$RECOVERIES/meta/info.json" ] || die "no recoveries at $RECOVERIES"
[ -f "$CAMERA" ] || die "no calibrated camera at $CAMERA"

IFS=',' read -r -a entries <<< "$PAIRS"
picks=() places=() weights=() total=0
for e in "${entries[@]}"; do
  IFS=: read -r pick place w <<< "$e"
  [ -n "$pick" ] && [ -n "$place" ] || die "entry '$e' is not pick:place[:weight]"
  picks+=("$pick") places+=("$place") weights+=("${w:-1}"); total=$((total + ${w:-1}))
done
[ "$total" -gt 0 ] || die "every weight in PAIRS=$PAIRS is 0"
per_pair_shards=$(( SHARDS / ${#entries[@]} )); [ "$per_pair_shards" -ge 1 ] || per_pair_shards=1
# the final save writes 12 GB of weights plus up to 36 GB of optimizer state; fail now, not after the queue wait
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 50 ] || die "only ${free_gb} GB free; this round needs about 50 for the final save"

if [ -f "$MIX/meta/info.json" ]; then say "mix already built: $MIX"; else
  pids=() names=()
  for i in "${!entries[@]}"; do
    pick=${picks[$i]} place=${places[$i]} w=${weights[$i]}
    each=$(( EPISODES * w / (total * per_pair_shards) )); [ "$each" -ge 1 ] || each=1
    say "recording $((each * per_pair_shards)) demos of $pick on $place (weight $w)"
    for k in $(seq 0 $((per_pair_shards - 1))); do
      name="${pick}_${place}_$k"; names+=("$name")
      rm -rf "${RAW:?}/$name"
      OMP_NUM_THREADS=1 nice -n 10 uv run --quiet npai demos "$RAW/$name" --episodes "$each" --seed "$RANDOM" \
        --randomize --role-agnostic --pick "$pick" --place "$place" --camera "$CAMERA" --original-share 0 \
        > "$LOGS/demos_$name.log" 2>&1 &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p" || die "a demo shard failed; see $LOGS/demos_*.log"; done
  pids=()
  for name in "${names[@]}"; do
    ( $LR/python scripts/episode_stats.py "$RAW/$name" > /dev/null &&
      $LR/python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id "local/pair_$name" --root "$RAW/$name" \
        --push-to-hub false > "$LOGS/convert_$name.log" 2>&1 ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || die "a convert failed; see $LOGS/convert_*.log"; done
  rm -rf "$MIX"
  $LR/python - "$TWIN" "$RECOVERIES" "$RAW" "$MIX" "$TAG" "$UPWEIGHT" "$REPEAT" "${names[@]}" <<'PY' || exit 1
import json, sys
from pathlib import Path
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
twin, recoveries, raw, out, tag = sys.argv[1:6]
upweight, repeat = int(sys.argv[6]), int(sys.argv[7])
names = sys.argv[8:]
features = json.loads(Path(twin, "meta/info.json").read_text())["features"]
parts = []
for name in names:  # the converter adds per-feature keys the merge won't accept: adopt the base set's schema
    info_path = Path(raw, name, "meta/info.json")
    info_path.write_text(json.dumps({**json.loads(info_path.read_text()), "features": features}, indent=4))
    parts.append(LeRobotDataset(f"local/pair_{name}", root=f"{raw}/{name}"))
base, rec = LeRobotDataset("local/base", root=twin), LeRobotDataset("local/rec", root=recoveries)
# the policy's own training mix beside the refused pairs, so stacking is rehearsed while the pairs are learnt
merged = merge_datasets([base, *[rec] * upweight, *parts * repeat], f"local/pairs_{tag}", output_dir=out,
                        concatenate_videos=False)
print(f"mix: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")
PY
fi

NEW=$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model
if [ -f "$NEW/config.json" ]; then say "checkpoint already trained: $NEW"; else
autoresearch-wait --label "pair-round-$TAG" --gpu-free-gb 70 --ram-free-gb 20 --stable 120 --timeout 43200 || exit 1
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 50 ] || die "only ${free_gb} GB free; the final save needs about 50"
say "fine-tuning $STEPS steps at lr $LEARNING_RATE"
rm -rf "$OUT"; ln -sfn "$VLM" "$POLICY/vlm_processor"
# a 4 h run that logs once an epoch is indistinguishable from a hang: supervise it, detached, PPID=1
setsid nohup autoresearch-heartbeat --label "pair-round-$TAG" --pattern lerobot-train --log "$LOGS/train.log" \
  </dev/null >> "$LOGS/heartbeat.log" 2>&1 & disown
(cd "$POLICY" && WANDB_PROJECT=preflight WANDB_RUN_GROUP=pairs $LR/lerobot-train \
  --dataset.repo_id="local/pairs_$TAG" --dataset.root="$MIX" \
  --policy.path="$POLICY" --policy.push_to_hub=false --policy.max_steps="$STEPS" --policy.optimizer_lr="$LEARNING_RATE" \
  --rename_map='{"observation.images.front": "observation.images.camera_0", "observation.images.wrist": "observation.images.camera_1"}' \
  --steps="$STEPS" --batch_size=8 --save_freq="$STEPS" --log_freq=20 --num_workers=8 --env_eval_freq=0 \
  --output_dir="$OUT" --job_name="pairs-$TAG" \
  --job.tags="[\"pairs\",\"gr00t-n1.7\",\"lerobot\",\"sim-only\"]" \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=preflight \
  --wandb.notes="$POLICY fine-tuned $STEPS steps (lr $LEARNING_RATE) on its own training mix plus demos of the refused pairs $PAIRS (x$REPEAT). Trained on what was refused, so a gain here closes those refusals and is not transfer." \
  > "$LOGS/train.log" 2>&1) || die "training failed; see $LOGS/train.log"
find "$OUT" -name training_state -type d -exec rm -rf {} + 2>/dev/null
fi

ln -sfn "$VLM" "$NEW/vlm_processor"
$LR/python scripts/lerobot_policy_server.py "$NEW" --port 5572 > "$LOGS/serve.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q serving "$LOGS/serve.log"; do kill -0 $SERVER 2>/dev/null || die "server died"; sleep 5; done
say "six stacking suites"
only=(); for s in "${STACK_SUITES[@]}"; do only+=(--only "$s"); done
uv run --quiet python scripts/benchmark.py run "pairs-$TAG" --port 5572 "${only[@]}" \
  --out "$LOGS/bench.json" --wandb --wandb-group pairs || die "benchmark failed"
uv run --quiet python scripts/benchmark.py compare "$REF_STACK" "$LOGS/bench.json"
say "PAIR_ROUND_DONE: $LOGS/bench.json"
