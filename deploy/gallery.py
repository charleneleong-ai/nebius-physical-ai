"""Render the demo gallery: synthetic phone frames of the A3 mat, each run through the job image on a GPU.

Each scene is a frame `tests/synthetic_scene.py` draws (the four corner markers, coloured blocks, perspective), run with
`preflight-job` exactly as an upload would be: Token Factory vision, the marker-fitted twin, rehearsals of the served
GR00T policy, the trust gate, and Nemotron's explanation. The gallery covers the verdicts a visitor should see: scenes
inside the arm's reach, a tight gap, the edge of reach, too close to the base, clutter, a hidden marker, and a task the
policy was never trained on.

    uv run python deploy/gallery.py frames OUT_DIR          # write the frames and scenes.json
    bash deploy/gallery.sh OUT_DIR CKPT_DIR                  # run each through the image (on the GPU host)
"""

import json
import sys
from pathlib import Path

import cv2
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from synthetic_scene import Block, scene

# The arm reaches 160-220 mm in front of its base, which sits 50 mm beyond the mat's bottom edge (y = 227 mm).
SCENES = {
    "centre": ((Block("blue", 150, 85), Block("red", 220, 85, 20)), "blue", "red", {}),
    "left": ((Block("blue", 100, 70), Block("red", 160, 100, 20)), "blue", "red", {}),
    "right": ((Block("blue", 260, 100), Block("red", 200, 70, 20)), "blue", "red", {}),
    "diagonal": ((Block("blue", 130, 110), Block("red", 210, 65, 20)), "blue", "red", {}),
    "reversed": ((Block("blue", 230, 80), Block("red", 140, 95, 20)), "blue", "red", {}),
    "tight_gap": ((Block("blue", 165, 85), Block("red", 210, 85, 20)), "blue", "red", {}),
    "edge_of_reach": ((Block("blue", 80, 50), Block("red", 270, 55, 20)), "blue", "red", {}),
    "too_close": ((Block("blue", 130, 150), Block("red", 220, 150, 20)), "blue", "red", {}),
    "cluttered": (
        (Block("blue", 150, 85), Block("red", 220, 85, 20), Block("yellow", 110, 115, 20), Block("dark", 260, 60, 18)),
        "blue",
        "red",
        {},
    ),
    "hidden_marker": ((Block("blue", 150, 85), Block("red", 220, 85, 20)), "blue", "red", {"markers": (0, 1, 2)}),
    "new_task": ((Block("yellow", 150, 85, 20), Block("red", 220, 85, 20)), "yellow", "red", {}),
}
app = typer.Typer(add_completion=False)


@app.command()
def frames(out: Path) -> None:
    """One frame per scene plus scenes.json (pick, place) for the runner."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (blocks, pick, place, options) in SCENES.items():
        cv2.imwrite(str(out / f"{name}.jpg"), scene(blocks, **options).image)
        manifest[name] = {"pick": pick, "place": place}
    (out / "scenes.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} frames -> {out}")


if __name__ == "__main__":
    app()
