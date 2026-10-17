#!/usr/bin/env bash
# Assemble the Space's build context (the package, both lockfiles' worth of metadata, the gallery) and upload it.
#   PYTHON=<a python with huggingface_hub> bash deploy/space/publish.sh GALLERY_DIR [SPACE_ID]
set -euo pipefail
gallery=$1 space=${2:-chaleong/preflight}
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
cp -r src pyproject.toml uv.lock README.md "$ctx"/
cp deploy/space/Dockerfile "$ctx"/Dockerfile
cp deploy/space/README.md "$ctx"/README.md  # the Space's own card (front matter sets the Docker SDK and port)
mkdir "$ctx/gallery"
shopt -s nullglob
for scene in "$gallery"/*/result.json; do
  name=$(basename "$(dirname "$scene")")
  [ -f "$gallery/$name.jpg" ] || { echo "no frame $gallery/$name.jpg for $name"; exit 1; }
  mkdir "$ctx/gallery/$name"
  cp "$scene" "$gallery/$name"/*.mp4 "$ctx/gallery/$name/"
  cp "$gallery/$name.jpg" "$ctx/gallery/"
done
[ -f "$gallery/suites.json" ] && cp -r "$gallery/suites.json" "$gallery/suites" "$ctx/gallery/"
find "$ctx" -name __pycache__ -prune -exec rm -rf {} +
# huggingface_hub's API rather than the `hf` CLI, whose flags differ across versions
${PYTHON:-python3} - "$space" "$ctx" <<'PY'
import sys
from huggingface_hub import create_repo, upload_folder
space, ctx = sys.argv[1:3]
create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
upload_folder(repo_id=space, repo_type="space", folder_path=ctx, commit_message="Preflight demo")
PY
echo "https://huggingface.co/spaces/$space"
