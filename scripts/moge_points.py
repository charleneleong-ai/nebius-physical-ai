"""MoGe-3 metric point maps for frames of a clip, for `npai lift`.

Runs in MoGe's own environment (Linux + CUDA; MoGe-3 needs Triton/FlexGEMM), not the project's:
    cd ~/moge && uv pip install --python .venv/bin/python typer
    cd ~/moge && .venv/bin/python ~/nebius-physical-ai/scripts/moge_points.py CLIP.mp4 points.npz --hfov 68.8
"""

from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
from moge.model.v3 import MoGeModel
import typer

app = typer.Typer()


@app.command()
def main(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Argument(dir_okay=False)],
    hfov: Annotated[float, typer.Option(help="The clip's horizontal FOV in degrees (npai calibrate-camera).")],
    frames: Annotated[int, typer.Option(min=1, show_default=True)] = 26,
    model: Annotated[str, typer.Option(show_default=True)] = "Ruicheng/moge-3-vitl",
) -> None:
    net = MoGeModel.from_pretrained(model).cuda().eval()
    cap = cv2.VideoCapture(str(video))
    indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frames).astype(int)
    points, valid = [], []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise typer.BadParameter(f"can't read frame {i} of {video}")
        rgb = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pred = net.infer(rgb.permute(2, 0, 1), fov_x=hfov)
        points.append((pred["points"].cpu().numpy() * 1000).astype(np.float16))  # camera frame, mm
        valid.append(pred["mask"].cpu().numpy() > 0.5)
    np.savez_compressed(out, indices=indices, points=np.stack(points), valid=np.stack(valid), hfov=hfov)
    typer.echo(f"wrote {len(indices)} point maps to {out}")


if __name__ == "__main__":
    app()
