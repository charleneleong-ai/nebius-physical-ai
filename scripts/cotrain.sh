#!/usr/bin/env bash
# Co-training replay (scripts/real_mix.py has why): v5's recipe exactly (hqfang base restatted on the training set,
# 12k steps of 8 at lr 2.5e-5) with ArmnetBench's real teleoperation mixed in at REAL_SHARE of the samples, then the
# six paired suites and so101-benchmark against v5. Conversion, mixing and restatting run right away; training waits
# for WAIT_FOR (a shell condition, e.g. DAgger round 3 finishing) to free the GPU.
#   WAIT_FOR='grep -qsE "round 3: (DAGGER_ROUND_DONE|.*failed)" ~/nebius-physical-ai/logs/dagger_r3_*.log' \\
#     scripts/detach.sh cotrain bash scripts/cotrain.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" MUJOCO_GL=egl
set -a; . ./.env; set +a

LR=~/lerobot-n17/.venv/bin
NAME=${NAME:-v9-cotrain} REAL_SHARE=${REAL_SHARE:-0.25} STEPS=${STEPS:-12000} LEARNING_RATE=${LEARNING_RATE:-2.5e-5}
WAIT_FOR=${WAIT_FOR:-true}
SUITES=(clean_many seen_vocabulary held_out_objects clean_top_down_many real_1_pomade_tin_many in_path_pomade_tin_many)
TWIN=$HOME/data/preflight_demos_v4c_v30 REAL=$HOME/data/armnet_twin MIX=$HOME/data/${NAME}_mix
EXTRA=${EXTRA:-}  # another converted real set mixed in, e.g. ~/data/armnet_rollouts for v9b
BASE=$HOME/checkpoints/hqfang-groot-so100_101 RESTAT=$HOME/checkpoints/hqfang-restat-$NAME
OUT=$HOME/checkpoints/gr00t-preflight-$NAME VLM=$BASE/vlm_processor
LOGS=$PWD/logs/$NAME
EXTERNAL=${EXTERNAL:-$HOME/npai-flux/scripts/so101_bench_eval.py}  # from #14, not on this branch
REF_BENCH=${REF_BENCH:-$HOME/nebius-physical-ai/logs/bench_v5_failures.json}  # v5 on the same six suites
mkdir -p "$LOGS"
say() { echo "[$(date -u +%T)] $NAME: $*"; }
die() { say "$*"; exit 1; }
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null' EXIT

if [ ! -f "$REAL/meta/info.json" ]; then
  say "converting ArmnetBench teleoperation"
  nice -n 10 $LR/python scripts/real_mix.py convert "$REAL" --like "$TWIN" > "$LOGS/convert.log" 2>&1 || die "convert failed"
fi
$LR/python scripts/real_mix.py check "$REAL" "$TWIN" || die "unit check failed"
say "mixing at real share $REAL_SHARE"
rm -rf "$MIX"
$LR/python scripts/real_mix.py mix "$TWIN" "$REAL" "$MIX" --real-share "$REAL_SHARE" ${EXTRA:+--extra "$EXTRA"} || die "mix failed"
if ! ls "$BASE"/model*.safetensors >/dev/null 2>&1; then
  say "re-downloading the base weights"
  hf download hqfang/groot-so100_101 --local-dir "$BASE" > "$LOGS/download.log" 2>&1 || die "download failed"
fi
rm -rf "$RESTAT"
$LR/python scripts/restat_checkpoint.py "$BASE" "$MIX" "$RESTAT" || die "restat failed"

until eval "$WAIT_FOR"; do sleep 120; done
# The final save writes about 12 GB of weights plus up to 36 GB of optimizer state before it is pruned.
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 50 ] || die "only ${free_gb} GB free; the final save needs about 50"

say "fine-tuning $STEPS steps from $RESTAT on $MIX"
rm -rf "$OUT"
# v5's schedule exactly: a 10k-step cosine horizon run for 12k steps.
(cd "$RESTAT" && WANDB_RUN_GROUP=cotrain $LR/lerobot-train \
  --dataset.repo_id="local/$(basename "$MIX")" --dataset.root="$MIX" \
  --policy.path="$RESTAT" --policy.push_to_hub=false --policy.max_steps=10000 --policy.optimizer_lr="$LEARNING_RATE" \
  --rename_map='{"observation.images.front": "observation.images.camera_0", "observation.images.wrist": "observation.images.camera_1"}' \
  --steps="$STEPS" --batch_size=8 --save_freq="$STEPS" --log_freq=20 --num_workers=8 --env_eval_freq=0 \
  --output_dir="$OUT" --job_name="$NAME" \
  --job.tags='["cotrain","gr00t-n1.7","lerobot","from:hqfang","real:armnetbench"]' \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=preflight \
  --wandb.notes="v5's recipe (hqfang base, 800 v4c twin demos, $STEPS steps of 8, lr $LEARNING_RATE) with ArmnetBench's 400 real teleoperated SO-101 demos mixed in at about $REAL_SHARE of the samples. Compare paired against v5." \
  > "$LOGS/train.log" 2>&1) || die "training failed; see $LOGS/train.log"
find "$OUT" -name training_state -type d -exec rm -rf {} + 2>/dev/null
NEW=$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model
ln -sfn "$VLM" "$NEW/vlm_processor"

say "benchmarking"
$LR/python scripts/lerobot_policy_server.py "$NEW" --port 5566 > "$LOGS/serve.log" 2>&1 &
SERVER=$!
until grep -q serving "$LOGS/serve.log"; do kill -0 $SERVER 2>/dev/null || die "server died"; sleep 10; done
uv run python scripts/benchmark.py run "$NAME" --port 5566 $(printf -- '--only %s ' "${SUITES[@]}") --out "$LOGS/bench.json" \
  --wandb --wandb-group cotrain || die "benchmark failed"
uv run python scripts/benchmark.py compare "$LOGS/bench.json" "$REF_BENCH"
uv run python scripts/benchmark.py failures "$LOGS/bench.json"
(cd ~/so101-benchmark && .venv/bin/python "$EXTERNAL" --port 5566 --episodes 50 --out "$LOGS/so101bench.json") \
  || say "so101-benchmark failed"
say "COTRAIN_DONE"
