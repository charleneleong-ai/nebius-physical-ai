#!/usr/bin/env bash
# Run every gallery frame (deploy/gallery.py) through the job image: OUT/<scene>/result.json and its rollout videos.
#   bash deploy/gallery.sh OUT_DIR CKPT_DIR [N]
set -uo pipefail
out=$(realpath "$1") ckpt=$(realpath "$2") n=${3:-32}
# saved checkpoints link the VLM processor from the base model's folder; mount the link's target so it resolves inside
vlm=()
[ -L "$ckpt/vlm_processor" ] && vlm=(-v "$(realpath "$ckpt/vlm_processor")":/models/policy/vlm_processor:ro)
for frame in "$out"/*.jpg; do
  name=$(basename "$frame" .jpg)
  read -r pick place < <(python3 -c "import json,sys; s=json.load(open('$out/scenes.json'))['$name']; print(s['pick'], s['place'])")
  [ -f "$out/$name/result.json" ] && { echo "$name: done"; continue; }
  echo "[$(date -u +%T)] $name ($pick on $place)"
  docker run --rm --gpus all -v "$ckpt":/models/policy:ro "${vlm[@]}" -v "$out":/work -e NEBIUS_API_KEY -e NEBIUS_PROJECT_ID \
    preflight-job:dev "/work/$name.jpg" "/work/$name" --n "$n" --pick "$pick" --place "$place" 2>&1 | tail -1
done
echo "GALLERY_DONE"
