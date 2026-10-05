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
import typer
from moge.model.v3 import MoGeModel

app = typer.Typer()


class MoGePointMapGenerator:
    """Generate metric point maps for evenly sampled video frames."""

    def __init__(self, model: str) -> None:
        self.model = MoGeModel.from_pretrained(model).cuda().eval()

    def generate(self, video: Path, out: Path, hfov: float, frames: int) -> None:
        capture = cv2.VideoCapture(str(video))
        try:
            indices = np.linspace(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frames).astype(int)
            points, valid = [], []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise typer.BadParameter(f"can't read frame {index} of {video}")
                rgb = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    prediction = self.model.infer(rgb.permute(2, 0, 1), fov_x=hfov)
                points.append((prediction["points"].cpu().numpy() * 1000).astype(np.float16))
                valid.append(prediction["mask"].cpu().numpy() > 0.5)
        finally:
            capture.release()
        np.savez_compressed(out, indices=indices, points=np.stack(points), valid=np.stack(valid), hfov=hfov)
        typer.echo(f"wrote {len(indices)} point maps to {out}")


@app.command()
def main(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Argument(dir_okay=False)],
    hfov: Annotated[float, typer.Option(help="The clip's horizontal FOV in degrees (npai calibrate-camera).")],
    frames: Annotated[int, typer.Option(min=1, show_default=True)] = 26,
    model: Annotated[str, typer.Option(show_default=True)] = "Ruicheng/moge-3-vitl",
) -> None:
    """Create point maps for a video clip."""
    MoGePointMapGenerator(model).generate(video, out, hfov, frames)


if __name__ == "__main__":
    app()
