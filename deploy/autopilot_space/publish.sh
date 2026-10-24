#!/usr/bin/env bash
# Upload a built Autopilot page (build.py OUT) as a static Space.
#   PYTHON=<a python with huggingface_hub> bash deploy/autopilot_space/publish.sh OUT [SPACE_ID]
set -euo pipefail
out=$1 space=${2:-chaleong/autopilot}
bash deploy/upload_space.sh "$space" "$out" static "Autopilot"
