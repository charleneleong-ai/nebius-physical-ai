"""Which `bbox_2d` convention does the vision model use?

For each frame the ArUco markers and the colour segmentation give every object's centre in pixels (ground truth that
does not depend on the vision model). The model's raw boxes are then read under each candidate convention and matched
to the nearest object; the convention with the smallest error wins.

    uv run python scripts/verify_bbox.py frame1.jpg frame2.png [--tf-vl]

Uses Cosmos (COSMOS_ENDPOINT_URL) unless --tf-vl. Exit code 1 when 0-1000 is not the best convention on every frame,
or when no frame had markers.
"""

import asyncio
import base64
import math
import re
import statistics
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer

from nebius_physical_ai.capture.mat import ScenePerceiver
from nebius_physical_ai.perception import RIG_WORDS, decode_frame, vl_jpeg_b64
from nebius_physical_ai.pipeline import CosmosClient, TokenFactoryVLClient

CONVENTIONS = {
    "0-1000 (x1,y1,x2,y2)": lambda b, w, h, sw, sh: ((b[0] + b[2]) / 2 / 1000 * w, (b[1] + b[3]) / 2 / 1000 * h),
    "0-1 (x1,y1,x2,y2)": lambda b, w, h, sw, sh: ((b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h),
    "pixels of the image sent": lambda b, w, h, sw, sh: ((b[0] + b[2]) / 2 * w / sw, (b[1] + b[3]) / 2 * h / sh),
    "0-1000 (y1,x1,y2,x2)": lambda b, w, h, sw, sh: ((b[1] + b[3]) / 2 / 1000 * w, (b[0] + b[2]) / 2 / 1000 * h),
}
EXPECTED = "0-1000 (x1,y1,x2,y2)"

app = typer.Typer(add_completion=False)


def truth_centres_px(
    image: np.ndarray, perceiver: ScenePerceiver
) -> tuple[list[tuple[str, tuple[float, float]]], float]:
    """Object centres in frame pixels from the markers and segmentation, and the mm per pixel at the mat's centre."""
    observation, location = perceiver.perceive(image)
    if location is None:
        return [], 0.0
    inverse = np.linalg.inv(location.homography)
    ppm = location.px_per_mm
    centres = []
    for obj in observation.objects:
        pt = cv2.perspectiveTransform(np.array([[[obj.x_mm * ppm, obj.y_mm * ppm]]], np.float32), inverse)[0, 0]
        centres.append((obj.label, (float(pt[0]), float(pt[1]))))
    mid = location.to_mat_mm([(image.shape[1] / 2, image.shape[0] / 2), (image.shape[1] / 2 + 1, image.shape[0] / 2)])
    return centres, float(np.linalg.norm(mid[1] - mid[0]))


def usable_boxes(objects: list[dict]) -> list[tuple[str, list[float]]]:
    boxes = []
    for raw in objects:
        label = str(raw.get("label", "")).lower()
        try:
            box = [float(v) for v in raw["bbox_2d"]]
        except (KeyError, TypeError, ValueError):
            continue
        if len(box) == 4 and not RIG_WORDS & set(re.split(r"\W+", label)):
            boxes.append((label, box))
    return boxes


def errors(boxes, truth, image, sent) -> dict[str, list[float]]:
    h, w = image.shape[:2]
    sh, sw = sent.shape[:2]
    result = {}
    for name, centre in CONVENTIONS.items():
        errs = []
        for _, box in boxes:
            cx, cy = centre(box, w, h, sw, sh)
            errs.append(min(math.hypot(cx - tx, cy - ty) for _, (tx, ty) in truth))
        result[name] = errs
    return result


async def raw_scene(use_token_factory: bool, image: np.ndarray):
    # One client per asyncio.run: an httpx client is bound to the event loop it was first used in.
    client = TokenFactoryVLClient() if use_token_factory else CosmosClient()
    try:
        return await client.perceive(vl_jpeg_b64(image))
    finally:
        await client.close()


def check_frame(path: Path, perceiver: ScenePerceiver, use_token_factory: bool) -> bool | None:
    """True when 0-1000 (x1, y1, x2, y2) fits best on this frame, False when another reading does or the model gave no
    usable boxes, None when the markers were not found (the frame says nothing about the model)."""
    image = decode_frame(base64.b64encode(path.read_bytes()).decode())
    truth, mm_per_px = truth_centres_px(image, perceiver)
    if not truth:
        print(f"{path.name}: markers not found, skipped")
        return None
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(vl_jpeg_b64(image)), np.uint8), cv2.IMREAD_COLOR)
    boxes = usable_boxes(asyncio.run(raw_scene(use_token_factory, image)).objects)
    print(
        f"\n{path.name}: {image.shape[1]}x{image.shape[0]} px (sent {sent.shape[1]}x{sent.shape[0]}), "
        f"{len(truth)} objects by markers, {len(boxes)} boxes from the model, {mm_per_px:.2f} mm/px"
    )
    for label, box in boxes:
        print(f"  model: {label:<14} bbox_2d={box}")
    if not boxes:
        print("  no usable boxes")
        return False
    errs = errors(boxes, truth, image, sent)
    best = min(errs, key=lambda k: statistics.median(errs[k]))
    for name, e in errs.items():
        med = statistics.median(e)
        print(
            f"  {'*' if name == best else ' '} {name:<26} median {med:7.1f} px = {med * mm_per_px:7.1f} mm, "
            f"worst {max(e):7.1f} px"
        )
    return best == EXPECTED


@app.command()
def main(
    frames: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="Frames with the mat's markers")],
    tf_vl: Annotated[bool, typer.Option("--tf-vl", help="Ask Token Factory VL instead of Cosmos")] = False,
) -> None:
    """Check which bbox_2d convention the vision model uses, against marker ground truth."""
    perceiver = ScenePerceiver()
    results = [check_frame(path, perceiver, tf_vl) for path in frames]
    checked = [r for r in results if r is not None]
    confirmed = bool(checked) and all(checked)
    print("\nconvention confirmed: 0-1000, (x1, y1, x2, y2)" if confirmed else "\nconvention NOT confirmed")
    raise typer.Exit(0 if confirmed else 1)


if __name__ == "__main__":
    app()
