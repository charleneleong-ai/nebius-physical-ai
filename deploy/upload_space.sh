#!/usr/bin/env bash
# Upload a built directory to a Hugging Face Space, creating the Space if it does not exist yet.
# Shared by the three publish.sh scripts, which differ only in what they build and which SDK serves it.
#   PYTHON=<a python with huggingface_hub> bash deploy/upload_space.sh SPACE_ID FOLDER SDK COMMIT_MESSAGE
set -euo pipefail
space=$1 folder=$2 sdk=$3 message=$4
# huggingface_hub's API rather than the `hf` CLI, whose flags differ across versions
${PYTHON:-python3} - "$space" "$folder" "$sdk" "$message" <<'PY'
import sys

from huggingface_hub import create_repo, upload_folder

space, folder, sdk, message = sys.argv[1:5]
create_repo(space, repo_type="space", space_sdk=sdk, exist_ok=True)
upload_folder(repo_id=space, repo_type="space", folder_path=folder, commit_message=message)
PY
echo "https://huggingface.co/spaces/$space"
