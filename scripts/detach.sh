#!/usr/bin/env bash
# Run a command fully detached (setsid, PPID=1) with a timestamped log; prints the log path.
#   scripts/detach.sh <name> <command...>
set -euo pipefail
name=$1
shift
mkdir -p logs
log=logs/${name}_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup "$@" </dev/null >>"$log" 2>&1 &
disown
echo "$log"
