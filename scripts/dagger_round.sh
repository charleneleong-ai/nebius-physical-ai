#!/usr/bin/env bash
# One DAgger round in the twin (scripts/dagger.py has the why). State lives in logs/dagger/state.env:
#   POLICY     served checkpoint (a LeRobot pretrained_model dir)
#   DATA       aggregate training set so far (LeRobot v3.0); with BASE set, the base demos alone
#   BASE       optional: the base demos; when set, recoveries are kept apart in RECOVERIES and upweighted
#   RECOVERIES optional: every round's recoveries so far (LeRobot v3.0), empty before the first
#   REF_BENCH  POLICY's result file on the gate suites
#   ROUND      last finished round
# A round: serve POLICY and rehearse fresh training scenes VARIANTS times each; keep up to BUDGET failures from scenes
# whose success rate is in BAND (most uncertain first, balanced across failure types); record the expert's recoveries
# and add them to DATA (always: the aggregate grows even when a round is not promoted); fine-tune POLICY on the
# aggregate with a short low-LR cosine schedule; benchmark the candidate and promote it only when `benchmark.py gate`
# passes.
#   bash scripts/dagger_round.sh          # one round;  for r in 1 2 3; do bash scripts/dagger_round.sh || break; done
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" MUJOCO_GL=egl
set -a; . ./.env; set +a
STATE=logs/dagger/state.env
# shellcheck disable=SC1090
. "$STATE" || { echo "write $STATE first (POLICY, DATA, REF_BENCH, ROUND=0)"; exit 1; }
LR=~/lerobot-n17/.venv/bin
VLM=$HOME/checkpoints/hqfang-groot-so100_101/vlm_processor
STAGE=${STAGE:-harvest}  # harvest | aggregate | train: resume a failed round from that stage
SCENES=${SCENES:-200} VARIANTS=${VARIANTS:-4} BUDGET=${BUDGET:-300} BAND=${BAND:-0.05,0.8}
PARTS=${PARTS:-8} STEPS=${STEPS:-4000} LEARNING_RATE=${LEARNING_RATE:-1e-5}
# With BASE set, training sees the base demos plus UPWEIGHT copies of all recoveries, so a few dozen recoveries are not
# drowned out by thousands of clean demos (round 1: 85 of 2,485 episodes, 3.5%, and no measurable change).
BASE=${BASE:-} RECOVERIES=${RECOVERIES:-} UPWEIGHT=${UPWEIGHT:-8}
TRAIN_WHEN=${TRAIN_WHEN:-true}  # shell condition polled before training, e.g. to wait for another job's GPU
SUITES=(clean_many seen_vocabulary held_out_objects clean_top_down_many real_1_pomade_tin_many in_path_pomade_tin_many)
R=$((ROUND + 1))
DIR=$PWD/logs/dagger/r$R RAW=$HOME/data/dagger_r$R NEXT_DATA=$HOME/data/dagger_agg_r${R}_v30
NEXT_RECOVERIES=${BASE:+$HOME/data/dagger_recoveries_r$R}
CAND=$HOME/checkpoints/gr00t-preflight-dagger-r$R
mkdir -p "$DIR"
say() { echo "[$(date -u +%T)] round $R: $*"; }
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null' EXIT  # never leave a policy server holding its port

serve() {  # serve CHECKPOINT PORT LOG -> sets SERVER
  ln -sfn "$VLM" "$1/vlm_processor"
  $LR/python scripts/lerobot_policy_server.py "$1" --port "$2" > "$3" 2>&1 &
  SERVER=$!
  until grep -q serving "$3"; do kill -0 $SERVER 2>/dev/null || { say "server died: $3"; exit 1; }; sleep 10; done
}

if [ "$STAGE" = harvest ]; then
  say "harvesting $SCENES scenes with $POLICY"
  serve "$POLICY" 5564 "$DIR/serve_harvest.log"
  uv run python scripts/dagger.py harvest "$DIR/snapshots.jsonl" --port 5564 --scenes "$SCENES" --seed $((900 + 10 * R)) \
    --variants "$VARIANTS" --budget "$BUDGET" --band "$BAND" \
    || exit 1
  kill $SERVER; SERVER=""

  say "recording recoveries in $PARTS parts"
  pids=()
  for k in $(seq 0 $((PARTS - 1))); do
    rm -rf "$RAW/part_$k"
    uv run python scripts/dagger.py recover "$DIR/snapshots.jsonl" "$RAW/part_$k" --part "$k" --parts "$PARTS" \
      > "$DIR/recover_$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { say "a recovery part failed; see $DIR/recover_*.log"; exit 1; }; done
  grep -h recovered "$DIR"/recover_*.log

fi  # STAGE=harvest

if [ "$STAGE" != train ]; then
say "aggregating into $NEXT_DATA"
for k in $(seq 0 $((PARTS - 1))); do
  grep -q '"codebase_version": "v3.0"' "$RAW/part_$k/meta/info.json" && continue  # converted by an earlier attempt
  $LR/python scripts/episode_stats.py "$RAW/part_$k" > /dev/null || exit 1
  $LR/python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id "local/dagger_r${R}_$k" --root "$RAW/part_$k" \
    --push-to-hub false > "$DIR/convert_$k.log" 2>&1 || { say "convert $k failed"; exit 1; }
done
rm -rf "$NEXT_DATA" ${NEXT_RECOVERIES:+"$NEXT_RECOVERIES"}
$LR/python - "$DATA" "$RAW" "$PARTS" "$R" "$NEXT_DATA" "$BASE" "$RECOVERIES" "$UPWEIGHT" "$NEXT_RECOVERIES" <<'PY' || exit 1
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

data, raw, parts, r, out, base, recoveries, upweight, next_recoveries = sys.argv[1:10]
parts = int(parts)
# The v2.1 -> v3.0 converter writes extra per-feature keys (fps, video encoder settings) that a merged set lacks, and
# merging insists the feature dicts match exactly; the data are the same, so the parts adopt the aggregate's schema.
features = json.loads(Path(base or data, "meta/info.json").read_text())["features"]
for k in range(parts):
    info_path = Path(raw, f"part_{k}", "meta/info.json")
    info = json.loads(info_path.read_text())
    assert {n: f["shape"] for n, f in info["features"].items()} == {n: f["shape"] for n, f in features.items()}
    info_path.write_text(json.dumps({**info, "features": features}, indent=4))
new = [LeRobotDataset(f"local/dagger_r{r}_{k}", root=f"{raw}/part_{k}") for k in range(parts)]
if not base:  # plain DAgger: the aggregate simply grows
    merged = merge_datasets([LeRobotDataset("local/aggregate", root=data), *new], f"local/dagger_agg_r{r}", output_dir=out)
else:  # recoveries kept apart and upweighted against the base demos
    earlier = [LeRobotDataset("local/recoveries", root=recoveries)] if recoveries else []
    kept = merge_datasets([*earlier, *new], f"local/dagger_recoveries_r{r}", output_dir=next_recoveries)
    base_set = LeRobotDataset("local/base", root=base)
    merged = merge_datasets([base_set] + [kept] * int(upweight), f"local/dagger_agg_r{r}", output_dir=out)
    print(f"recoveries: {kept.meta.total_episodes} episodes, x{upweight} against {base_set.meta.total_episodes} base")
print(f"aggregate: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")
PY
rm -rf "$RAW"
fi  # STAGE!=train

until eval "$TRAIN_WHEN"; do sleep 120; done
# The final save writes about 12 GB of weights plus up to 36 GB of optimizer state before it is pruned.
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 50 ] || { say "only ${free_gb} GB free; the final save needs about 50"; exit 1; }
say "fine-tuning $STEPS steps at lr $LEARNING_RATE from $POLICY"
rm -rf "$CAND"
ln -sfn "$VLM" "$POLICY/vlm_processor"
(cd "$POLICY" && WANDB_PROJECT=preflight WANDB_RUN_GROUP=dagger $LR/lerobot-train \
  --dataset.repo_id="local/dagger_agg_r$R" --dataset.root="$NEXT_DATA" \
  --policy.path="$POLICY" --policy.push_to_hub=false --policy.max_steps="$STEPS" --policy.optimizer_lr="$LEARNING_RATE" \
  --rename_map='{"observation.images.front": "observation.images.camera_0", "observation.images.wrist": "observation.images.camera_1"}' \
  --steps="$STEPS" --batch_size=8 --save_freq="$STEPS" --log_freq=20 --num_workers=8 --env_eval_freq=0 \
  --output_dir="$CAND" --job_name="dagger-r$R" \
  --job.tags="[\"dagger\",\"round:$R\",\"gr00t-n1.7\",\"lerobot\",\"from:$(basename "$(dirname "$(dirname "$POLICY")")")\",\"sim-only\"]" \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=preflight \
  --wandb.notes="DAgger round $R: $POLICY fine-tuned $STEPS steps (lr $LEARNING_RATE, cosine) on the aggregate plus the expert's recoveries from its own failure states on $SCENES fresh training scenes. Promoted only past benchmark.py gate against $REF_BENCH." \
  > "$DIR/train.log" 2>&1) || { say "training failed; see $DIR/train.log"; exit 1; }
find "$CAND" -name training_state -type d -exec rm -rf {} + 2>/dev/null
NEW=$CAND/checkpoints/$(printf %06d "$STEPS")/pretrained_model

say "benchmarking the candidate"
serve "$NEW" 5565 "$DIR/serve_bench.log"
only=(); for s in "${SUITES[@]}"; do only+=(--only "$s"); done
rm -f "$DIR/bench.json"  # a retried round must never be gated on the last attempt's scores
uv run python scripts/benchmark.py run "dagger-r$R" --port 5565 "${only[@]}" --out "$DIR/bench.json" \
  --wandb --wandb-group dagger 2>/dev/null | grep -E "^dagger" || { say "benchmark failed"; exit 1; }
kill $SERVER; SERVER=""
uv run python scripts/benchmark.py failures "$DIR/bench.json"

OLD_POLICY=$POLICY OLD_DATA=$DATA
if uv run python scripts/benchmark.py gate "$REF_BENCH" "$DIR/bench.json" | tee "$DIR/gate.txt"; then
  POLICY=$NEW REF_BENCH=$DIR/bench.json
fi
# The state is written before anything is deleted, so it never points at a removed checkpoint.
printf 'POLICY=%s\nDATA=%s\nREF_BENCH=%s\nROUND=%s\nBASE=%s\nRECOVERIES=%s\n' \
  "$POLICY" "${BASE:-$NEXT_DATA}" "$REF_BENCH" "$R" "$BASE" "${NEXT_RECOVERIES:-}" > "$STATE"
if [ "$POLICY" = "$NEW" ]; then
  case "$OLD_POLICY" in *gr00t-preflight-dagger-r*) rm -rf "${OLD_POLICY%/checkpoints/*}" ;; esac  # superseded round
else
  rm -rf "$CAND"
fi
[[ "$OLD_DATA" == *dagger_agg_r* ]] && rm -rf "$OLD_DATA"  # the new aggregate contains it
[[ -n "$BASE" && "$RECOVERIES" == *dagger_recoveries_r* ]] && rm -rf "$RECOVERIES"  # superseded by NEXT_RECOVERIES
[ -n "$BASE" ] && rm -rf "$NEXT_DATA"  # upweighted training copy: rebuilt from BASE and RECOVERIES every round
say "DAGGER_ROUND_DONE; serving next: $POLICY"
