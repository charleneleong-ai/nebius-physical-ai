#!/usr/bin/env bash
# Failure-state synthesis (CARE, arXiv 2609.24118) as a DAgger side round on round 3. DAgger's yield shrinks as the
# policy improves (rounds 3-5 kept 146, 120, 113 failures), so instead of a fresh harvest this seeds COPIES nearby states
# from each failure round 3 already reached (rounds 4 and 5's harvests, both run with round 3 served), lets the scripted
# expert record recoveries from all of them, and fine-tunes round 3 with round 3's own recoveries plus the new ones,
# upweighted so recoveries hold about the share they had in round 3. Runs on its own state file, so the main DAgger line
# (logs/dagger/state.env) still serves round 3, and a promotion here keeps round 3's checkpoint and recoveries.
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
echo 1000 > /proc/self/oom_score_adj
say() { echo "[$(date -u +%T)] synth: $*"; }
COPIES=${COPIES:-6} UPWEIGHT=${UPWEIGHT:-3}
MAIN=$HOME/nebius-physical-ai/logs/dagger
STATE=logs/dagger/state.synth.env DIR=$PWD/logs/dagger/r6s  # dagger_round.sh: round 5 + 1, TAG=s
mkdir -p "$DIR"
cat > "$STATE" <<STATE_FILE
POLICY=$HOME/checkpoints/gr00t-preflight-dagger-r3/checkpoints/004000/pretrained_model
DATA=$HOME/data/preflight_demos_v6_v30
REF_BENCH=$MAIN/r3/bench.json
ROUND=5
BASE=$HOME/data/preflight_demos_v6_v30
RECOVERIES=$HOME/data/dagger_recoveries_r3s
RL_RECOVERIES=
STATE_FILE
say "synthesising $COPIES states per failure from rounds 4 and 5"
uv run --quiet python scripts/dagger.py synthesise "$MAIN/r4/snapshots.jsonl" "$MAIN/r5/snapshots.jsonl" \
  --out "$DIR/snapshots.jsonl" --copies "$COPIES" || { say "synthesis failed"; exit 1; }
STATE=$STATE STAGE=recover TAG=s UPWEIGHT=$UPWEIGHT KEEP_SUPERSEDED=1 \
  TRAIN_WHEN="autoresearch-wait --label synth-train --gpu-free-gb 70 --ram-free-gb 20 --stable 120" \
  bash scripts/dagger_round.sh || { say "round failed"; exit 1; }
say "SYNTH_DONE"
