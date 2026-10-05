#!/usr/bin/env bash
# Host the Cosmos 3 Reasoner (vLLM, OpenAI-compatible) on this machine's GPU, next to the GR00T servers.
#   scripts/cosmos_local.sh up | down | status | logs [-f] | print
# The server listens on 127.0.0.1 only and has no auth, so run `npai preflight` on the same machine with
#   COSMOS_ENDPOINT_URL=http://127.0.0.1:8001  COSMOS_MODEL=cosmos
# Cosmos3-Nano is not gated: no Hugging Face token. First start pulls the ~19 GB image and ~35 GB of weights into
# $COSMOS_CACHE; `rm -rf` that directory to give the disk back.
set -euo pipefail

NAME=${COSMOS_NAME:-preflight-cosmos-local}
IMAGE=${COSMOS_IMAGE:-vllm/vllm-openai:cosmos3}  # vLLM >= 0.23 with native Cosmos 3 Reasoner support
MODEL=${COSMOS_MODEL_ID:-nvidia/Cosmos3-Nano}
SERVED=${COSMOS_SERVED_NAME:-cosmos}  # what clients put in `model`; keep in sync with SERVED_MODEL_NAME in the package
PORT=${COSMOS_PORT:-8001}
GPU_FRACTION=${COSMOS_GPU_FRACTION:-0.30}  # of the whole GPU: ~17 GB of weights plus KV cache, leaving room for GR00T
MAX_LEN=${COSMOS_MAX_MODEL_LEN:-16384}
CACHE=${COSMOS_CACHE:-$HOME/.cache/cosmos3}

run_command() {
  printf '%q ' docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
    -e HF_HOME=/hf -v "$CACHE:/hf" \
    -p "127.0.0.1:$PORT:8000" \
    "$IMAGE" "$MODEL" \
    --served-model-name "$SERVED" --async-scheduling \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_FRACTION" --port 8000
  echo
}

# `docker inspect` prints a blank line for a missing container, so ask `docker ps` instead: empty output means absent.
state() {
  local found
  found=$(docker ps -a --filter "name=^${NAME}\$" --format '{{.State}}')
  echo "${found:-absent}"
}

case "${1:-}" in
  print)
    run_command
    ;;
  up)
    case "$(state)" in
      running) echo "$NAME is already running" ;;
      absent)
        mkdir -p "$CACHE"
        eval "$(run_command)"
        echo "$NAME starting; first run downloads the image and weights. Check with: scripts/cosmos_local.sh status"
        ;;
      *) docker start "$NAME" >/dev/null && echo "$NAME restarted" ;;
    esac
    ;;
  down)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "$NAME removed" || echo "$NAME was not running"
    ;;
  status)
    echo "container: $(state)"
    if models=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); then
      echo "serving: $(printf '%s' "$models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
    else
      echo "not answering on 127.0.0.1:$PORT yet"
      exit 1
    fi
    ;;
  logs)
    shift
    if [ "${1:-}" = "-f" ]; then docker logs -f --tail 100 "$NAME"; else docker logs --tail 80 "$NAME"; fi
    ;;
  *)
    echo "usage: $0 up | down | status | logs [-f] | print" >&2
    exit 2
    ;;
esac
