#!/usr/bin/env bash
# Hold the last frame of every rollout clip under MIN_S seconds so it reaches MIN_S: a successful rollout ends the
# moment the block is placed, and a 3 s loop restarts before the viewer sees the outcome. Idempotent.
#   bash deploy/hold_clips.sh GALLERY_DIR      # after deploy/gallery.sh and deploy/suite_cards.py
set -euo pipefail
gallery=$1 min_s=${MIN_S:-6}
find "$gallery" -name '*.mp4' -print0 | while IFS= read -r -d '' clip; do
  duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$clip" </dev/null)
  hold=$(python3 -c "print(max(0.0, $min_s - $duration))")
  if python3 -c "import sys; sys.exit(0 if $hold > 0.05 else 1)"; then
    ffmpeg -nostdin -v error -y -i "$clip" -vf "tpad=stop_mode=clone:stop_duration=$hold" -c:v libx264 -pix_fmt yuv420p \
      -movflags +faststart "$clip.tmp.mp4" && mv "$clip.tmp.mp4" "$clip"
    echo "held $clip ${duration}s -> ${min_s}s"
  fi
done
