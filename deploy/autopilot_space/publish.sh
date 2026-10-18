#!/usr/bin/env bash
# Upload a built Autopilot page (build.py OUT) as a static Space.
#   PYTHON=<a python with huggingface_hub> bash deploy/autopilot_space/publish.sh OUT [SPACE_ID]
set -euo pipefail
out=$1 space=${2:-chaleong/autopilot}
${PYTHON:-python3} - "$space" "$out" <<'PY'
import sys
from huggingface_hub import create_repo, upload_folder
space, out = sys.argv[1:3]
create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
upload_folder(repo_id=space, repo_type="space", folder_path=out, commit_message="Autopilot")
PY
echo "https://huggingface.co/spaces/$space"
