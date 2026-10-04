"""MoGe-3 metric point maps for frames of a clip, for `npai lift`.

Runs in MoGe's own environment (Linux + CUDA; MoGe-3 needs Triton/FlexGEMM), not the project's:
    cd ~/moge && .venv/bin/python ~/nebius-physical-ai/scripts/moge_points.py CLIP.mp4 points.npz --hfov 68.8
"""

import click
import cv2
import numpy as np
import torch
from moge.model.v3 import MoGeModel


@click.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.argument("out", type=click.Path(dir_okay=False))
@click.option("--hfov", type=float, required=True, help="The clip's horizontal FOV in degrees (npai calibrate-camera).")
@click.option("--frames", default=26, show_default=True, help="Frames sampled evenly across the clip.")
@click.option("--model", default="Ruicheng/moge-3-vitl", show_default=True)
def main(video: str, out: str, hfov: float, frames: int, model: str) -> None:
    net = MoGeModel.from_pretrained(model).cuda().eval()
    cap = cv2.VideoCapture(video)
    indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frames).astype(int)
    points, valid = [], []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise click.ClickException(f"can't read frame {i} of {video}")
        rgb = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pred = net.infer(rgb.permute(2, 0, 1), fov_x=hfov)
        points.append((pred["points"].cpu().numpy() * 1000).astype(np.float16))  # camera frame, mm
        valid.append(pred["mask"].cpu().numpy() > 0.5)
    np.savez_compressed(out, indices=indices, points=np.stack(points), valid=np.stack(valid), hfov=hfov)
    click.echo(f"wrote {len(indices)} point maps to {out}")


if __name__ == "__main__":
    main()
