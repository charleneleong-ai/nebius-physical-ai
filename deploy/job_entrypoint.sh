#!/usr/bin/env bash
# Start the GR00T policy server on the mounted checkpoint, run Preflight on the frame, stop the server.
# Arguments go to `npai preflight-job` (FRAME OUT [--n N] ...); the server's log lands in OUT beside result.json.
set -uo pipefail
[ -d "$CKPT" ] || { echo "no checkpoint at CKPT=$CKPT"; exit 2; }
mkdir -p "$2" && log=$2/policy_server.log
/opt/lerobot/.venv/bin/python /app/scripts/lerobot_policy_server.py "$CKPT" --port 5555 > "$log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
trap 'exit 143' TERM INT  # bash as PID 1 ignores SIGTERM otherwise, so a cancelled job would wait for SIGKILL
for _ in $(seq 200); do  # up to 10 minutes for the checkpoint to load
  grep -qs serving "$log" && break
  kill -0 $server 2>/dev/null || { echo "policy server died:"; tail -20 "$log"; exit 3; }
  sleep 3
done
grep -qs serving "$log" || { echo "policy server not serving after 10 minutes"; exit 3; }
/app/.venv/bin/npai preflight-job "$@" --port 5555
