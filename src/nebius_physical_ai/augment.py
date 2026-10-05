"""Is an appearance-augmented demo video still the same demo?

Cosmos video-to-video can change colours, textures, lighting and background while edge or depth control holds the
geometry. The demo's actions only stay valid if the geometry really held: a block that moved a few millimetres in the
video is a block the recorded gripper no longer reaches. The twin renders no markers, so the check works on the pictures
themselves: edge maps (which ignore colour and lighting) of the original and augmented frames are compared tile by
tile with phase correlation, and the shift of each tile is the drift in pixels.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

GRID = (4, 4)  # tiles per image: rows, columns
MAX_DRIFT_PX = 4.0  # at 640x480 a 30 mm block is 40 to 60 px wide, so this is 2 to 3 mm on the mat
MAX_UNMATCHED = 0.25  # share of edge-rich tiles whose edges don't line up at all
MIN_RESPONSE = 0.08  # phase-correlation peak below which two tiles show different things


def edges(image: np.ndarray) -> np.ndarray:
    """Edge strength in [0, 1], independent of colour, brightness and contrast."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image, (0, 0), 1.5)
    gray = gray.astype(np.float32)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return magnitude / max(float(magnitude.max()), 1e-6)


@dataclass(frozen=True)
class Drift:
    """How far the augmented frames' geometry sits from the originals'."""

    median_px: float
    max_px: float
    unmatched: float  # share of compared tiles that did not line up at all
    tiles: int

    def acceptable(self, max_px: float = MAX_DRIFT_PX, max_unmatched: float = MAX_UNMATCHED) -> bool:
        return self.tiles > 0 and self.median_px <= max_px and self.unmatched <= max_unmatched


def tile_shifts(original: np.ndarray, augmented: np.ndarray) -> list[float | None]:
    """Shift in pixels of each edge-rich tile (None where the tile's edges don't match), for one frame pair."""
    a, b = edges(original), edges(augmented)
    rows, cols = GRID
    th, tw = a.shape[0] // rows, a.shape[1] // cols
    window = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    tiles = [(slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw)) for r in range(rows) for c in range(cols)]
    energy = np.array([a[t].mean() for t in tiles])
    shifts: list[float | None] = []
    for tile, e in zip(tiles, energy):
        if e < np.median(energy):  # flat tiles say nothing about where things are
            continue
        (dx, dy), response = cv2.phaseCorrelate(a[tile], b[tile], window)
        shifts.append(float(np.hypot(dx, dy)) if response >= MIN_RESPONSE else None)
    return shifts


def measure_drift(originals: Sequence[np.ndarray], augmented: Sequence[np.ndarray]) -> Drift:
    """Drift over matching frames (same index, same time in the episode). Frames must show the same field of view."""
    if len(originals) != len(augmented):
        raise ValueError(f"{len(originals)} original frames but {len(augmented)} augmented ones")
    shifts: list[float | None] = []
    for original, other in zip(originals, augmented):
        if other.shape[:2] != original.shape[:2]:
            other = cv2.resize(other, original.shape[1::-1], interpolation=cv2.INTER_AREA)
        shifts += tile_shifts(original, other)
    matched = [s for s in shifts if s is not None]
    return Drift(
        median_px=float(np.median(matched)) if matched else float("inf"),
        max_px=float(max(matched)) if matched else float("inf"),
        unmatched=(len(shifts) - len(matched)) / len(shifts) if shifts else 1.0,
        tiles=len(shifts),
    )
