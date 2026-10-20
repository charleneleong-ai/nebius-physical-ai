#!/usr/bin/env bash
# Publish docs/artifact.html (scripts/build_artifact.py) as a static Space: the page in a document skeleton, with
# Mermaid loaded from a CDN in place of the Claude artifact runtime.
#   PYTHON=<a python with huggingface_hub> bash deploy/experiments_space/publish.sh [SPACE_ID]
set -euo pipefail
space=${1:-chaleong/preflight-experiments}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
  printf '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
  cat docs/artifact.html
  printf '\n<script type="module">\nimport mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";\n'
  printf 'mermaid.initialize({ startOnLoad: true, theme: "dark" });\n</script>\n</body>\n</html>\n'
} > "$out/index.html"
cp deploy/experiments_space/README.md "$out/"
[ -f deploy/experiments_space/thumbnail.png ] && cp deploy/experiments_space/thumbnail.png "$out/"
${PYTHON:-python3} - "$space" "$out" <<'PY'
import sys
from huggingface_hub import create_repo, upload_folder
space, out = sys.argv[1:3]
create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
upload_folder(repo_id=space, repo_type="space", folder_path=out, commit_message="Preflight experiments")
PY
echo "https://huggingface.co/spaces/$space"
