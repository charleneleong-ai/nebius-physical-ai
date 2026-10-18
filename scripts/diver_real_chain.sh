#!/usr/bin/env bash
# DiVeR follow-up (#30) on real outcomes: ArmnetBench's 255 block-stacking episodes (50 teleop, 205 policy rollouts of
# which most failed; human-judged labels), joint-calibrated (real_mix.py convert --all-outcomes). The served round 3 draws
# its 8 candidates for every real decision point; each real step is labelled with its episode's outcome and trains the
# observation-and-chunk verifier beside the same 300 twin rollouts #30 used, so the change from #30 is the real data
# alone. Held-out AUC stays on twin episodes; best-of-8 is benchmarked on held_out_objects and clean_many vs round 3.
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl PYTHONPATH=$PWD/src
echo 1000 > /proc/self/oom_score_adj
say() { echo "[$(date -u +%T)] diver-real: $*"; }
CKPT=$HOME/checkpoints/gr00t-preflight-dagger-r3/checkpoints/004000/pretrained_model
REAL=$HOME/data/armnet_stack_all_cal SIM=$HOME/data/diver_r3 DATA=$HOME/data/diver_real OUT=$HOME/checkpoints/diver-r3-real
LOGS=$PWD/logs/diver-real REF=$HOME/nebius-physical-ai/logs/dagger/r3/bench.json
mkdir -p "$LOGS" "$OUT"
[ -f "$REAL/labels.json" ] || { say "no converted real episodes at $REAL"; exit 1; }
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null' EXIT
serve() {  # serve LOG ARGS...
  log=$1; shift
  ~/lerobot-n17/.venv/bin/python scripts/lerobot_policy_server.py "$CKPT" --port 5569 --samples 8 "$@" > "$log" 2>&1 &
  SERVER=$!
  until grep -q serving "$log"; do kill -0 $SERVER 2>/dev/null || { say "server died; see $log"; exit 1; }; sleep 5; done
}
autoresearch-wait --label diver-real --gpu-free-gb 40 --ram-free-gb 20 --stable 120 --timeout 86400 || exit 1
say "scoring the real episodes' decision points"
serve "$LOGS/serve_collect.log" --select first
~/lerobot-n17/.venv/bin/python scripts/diver.py collect-real "$REAL" "$DATA" --port 5569 || { say "collect-real failed"; exit 1; }
kill $SERVER; SERVER=""
say "training the verifier on twin + real outcomes"
~/lerobot-n17/.venv/bin/python scripts/diver.py train "$SIM" "$OUT/verifier.pt" --beta 1.0 --extra "$DATA" \
  || { say "train failed"; exit 1; }
say "benchmarking round 3 + real-outcome verifier best-of-8"
serve "$LOGS/serve_bench.log" --select verifier --verifier "$OUT/verifier.pt"
uv run --quiet python scripts/benchmark.py run dagger-r3-diver8-real --port 5569 --only held_out_objects \
  --only clean_many --batch-size 4 --out "$LOGS/bench.json" --wandb --wandb-group diver || { say "benchmark failed"; exit 1; }
kill $SERVER; SERVER=""
uv run --quiet python scripts/benchmark.py compare "$LOGS/bench.json" "$REF"
uv run --quiet python scripts/benchmark.py failures "$LOGS/bench.json"
say "DIVER_REAL_DONE"
