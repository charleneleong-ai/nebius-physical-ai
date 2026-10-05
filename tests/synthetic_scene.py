"""Synthetic phone frames: an A3 mat with its four ArUco markers and coloured blocks, seen in perspective.

The scene knows the true mapping from mat millimetres to frame pixels, so tests can place a "vision model" box exactly
where an object really is, or deliberately somewhere else, and check what the marker correction does with it.
"""

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from nebius_physical_ai.capture.mat import ARUCO_DICT, MatSpec
from nebius_physical_ai.twin.scene import A3_MAT

PX = 3  # synthetic top-down resolution, px per mm
MARGIN_MM = 40
BGR = {"blue": (220, 60, 0), "red": (0, 0, 220), "yellow": (0, 220, 230), "green": (0, 200, 0), "dark": (30, 30, 30)}


@dataclass(frozen=True)
class Block:
    label: str
    x_mm: float
    y_mm: float
    half_mm: float = 15.0


CLEAN = (Block("blue", 130, 150), Block("red", 220, 150, 20))
CLUTTERED = (*CLEAN, Block("yellow", 90, 60, 20), Block("dark", 290, 70, 18))


def mm(v: float) -> int:
    return round((v + MARGIN_MM) * PX)


@dataclass
class SyntheticScene:
    image: np.ndarray  # BGR frame
    matrix: np.ndarray  # top-down canvas px -> frame px

    def pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        point = cv2.perspectiveTransform(np.float32([[[mm(x_mm), mm(y_mm)]]]), self.matrix)[0, 0]
        return float(point[0]), float(point[1])

    def box(self, x_mm: float, y_mm: float, half_mm: float) -> list[float]:
        """bbox_2d of a square on the mat, normalised to 0-1000 as the vision model reports it."""
        corners = [self.pixel(x_mm + dx, y_mm + dy) for dx in (-half_mm, half_mm) for dy in (-half_mm, half_mm)]
        height, width = self.image.shape[:2]
        us, vs = zip(*corners)
        return [min(us) / width * 1000, min(vs) / height * 1000, max(us) / width * 1000, max(vs) / height * 1000]

    def vl_object(self, block: Block, confidence: float = 0.9, at: tuple[float, float] | None = None) -> dict:
        """What a vision model would report for `block`, optionally misplaced to `at` (mat mm)."""
        x, y = at or (block.x_mm, block.y_mm)
        return {
            "label": block.label,
            "bbox_2d": self.box(x, y, block.half_mm),
            "shape": "box",
            "confidence": confidence,
        }

    def b64(self) -> str:
        ok, jpeg = cv2.imencode(".jpg", self.image)
        assert ok
        return base64.b64encode(jpeg.tobytes()).decode()


def scene(
    blocks: tuple[Block, ...] = CLUTTERED,
    spec: MatSpec = A3_MAT,
    markers: tuple[int, ...] = (0, 1, 2, 3),
    skew: float = 0.12,
    marker_shift_mm: dict[int, tuple[float, float]] | None = None,
) -> SyntheticScene:
    """`marker_shift_mm` moves a printed marker off its corner, as a buckled or mis-measured sheet would."""
    size = mm(spec.width_mm + MARGIN_MM), mm(spec.height_mm + MARGIN_MM)
    top_down = np.full((size[1], size[0], 3), 235, np.uint8)
    half = round(spec.marker_mm * PX / 2)
    for i, (x, y) in spec.marker_centres_mm.items():
        if i in markers:
            dx, dy = (marker_shift_mm or {}).get(i, (0.0, 0.0))
            marker = cv2.aruco.generateImageMarker(ARUCO_DICT, i, 2 * half)
            cx, cy = mm(x + dx), mm(y + dy)
            top_down[cy - half : cy + half, cx - half : cx + half] = marker[..., None]
    for b in blocks:
        cv2.rectangle(
            top_down,
            (mm(b.x_mm - b.half_mm), mm(b.y_mm - b.half_mm)),
            (mm(b.x_mm + b.half_mm), mm(b.y_mm + b.half_mm)),
            BGR[b.label],
            -1,
        )
    h, w = top_down.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * skew, h * 0.05], [w * (1 - skew), 0], [w, h], [0, h * 0.95]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.GaussianBlur(cv2.warpPerspective(top_down, matrix, (w, h)), (3, 3), 0)
    return SyntheticScene(frame, matrix)
