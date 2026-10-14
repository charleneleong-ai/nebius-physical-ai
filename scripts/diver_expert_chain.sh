#!/usr/bin/env bash
# DiVeR follow-up (#30): the first verifier ranked held-out steps at AUC 0.79 yet picking with it was flat, plausibly
# because its candidates shared the observation features and differed only by the chunk. DiVeR scores the action
# expert's per-candidate hidden state; this collects round 3's rollouts again with that state recorded, trains the
# verifier on it, and benchmarks best-of-8 on held_out_objects and clean_many against round 3. Waits for the GPU.
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
echo 1000 > /proc/self/oom_score_adj
say() { echo "[$(date -u +%T)] diver-expert: $*"; }
CKPT=$HOME/checkpoints/gr00t-preflight-dagger-r3/checkpoints/004000/pretrained_model
DATA=$HOME/data/diver_r3_expert OUT=$HOME/checkpoints/diver-r3-expert LOGS=$PWD/logs/diver-expert
REF=$HOME/nebius-physical-ai/logs/dagger/r3/bench.json
mkdir -p "$LOGS" "$OUT"
ln -sfn "$HOME/checkpoints/hqfang-groot-so100_101/vlm_processor" "$CKPT/vlm_processor"
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null' EXIT
serve() {  # serve LOG ARGS...
  log=$1; shift
  ~/lerobot-n17/.venv/bin/python scripts/lerobot_policy_server.py "$CKPT" --port 5568 --samples 8 "$@" > "$log" 2>&1 &
  SERVER=$!
  until grep -q serving "$log"; do kill -0 $SERVER 2>/dev/null || { say "server died; see $log"; exit 1; }; sleep 5; done
}
autoresearch-wait --label diver-expert-collect --gpu-free-gb 40 --ram-free-gb 20 --stable 120 --timeout 86400 || exit 1
say "collecting into $DATA"
serve "$LOGS/serve_collect.log" --select first
uv run --quiet python scripts/diver.py collect "$DATA" --port 5568 --scenes 150 --variants 2 --batch-size 8 \
  || { say "collect failed"; exit 1; }
kill $SERVER; SERVER=""
say "training the verifier with action-expert state"
~/lerobot-n17/.venv/bin/python scripts/diver.py train "$DATA" "$OUT/verifier.pt" --beta 1.0 --expert \
  || { say "train failed"; exit 1; }
autoresearch-wait --label diver-expert-bench --gpu-free-gb 40 --ram-free-gb 20 --stable 120 || exit 1
say "benchmarking round 3 + expert-state verifier best-of-8"
serve "$LOGS/serve_bench.log" --select verifier --verifier "$OUT/verifier.pt"
uv run --quiet python scripts/benchmark.py run dagger-r3-diver8-expert --port 5568 --only held_out_objects \
  --only clean_many --batch-size 4 --out "$LOGS/bench.json" --wandb --wandb-group diver || { say "benchmark failed"; exit 1; }
kill $SERVER; SERVER=""
uv run --quiet python scripts/benchmark.py compare "$LOGS/bench.json" "$REF"
uv run --quiet python scripts/benchmark.py failures "$LOGS/bench.json"
say "DIVER_EXPERT_DONE"
