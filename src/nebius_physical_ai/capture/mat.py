from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cached_property, partial
from typing import Any

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CORNER_NAMES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}
MIN_MARKERS = 2

# OpenCV hue is 0-179; red wraps around so it gets two bands.
COLOUR_HUES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 8), (170, 179)],
    "yellow": [(20, 35)],
    "green": [(40, 85)],
    "blue": [(95, 130)],
}
OPEN_KERNEL = np.ones((5, 5), np.uint8)


@dataclass(frozen=True)
class MatSpec:
    """Marker centres sit on the corners of a width x height rectangle, markers upright."""

    width_mm: float = 300.0
    height_mm: float = 300.0
    marker_mm: float = 50.0
    px_per_mm: float = 2.0

    @cached_property
    def marker_centres_mm(self) -> dict[int, tuple[float, float]]:
        w, h = self.width_mm, self.height_mm
        return {0: (0.0, 0.0), 1: (w, 0.0), 2: (w, h), 3: (0.0, h)}

    @cached_property
    def board(self) -> cv2.aruco.Board:
        half = self.marker_mm / 2
        # ArUco corner order is clockwise from top-left; mat y points down, matching image y.
        offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
        obj_points = [
            np.array([(cx + dx, cy + dy, 0.0) for dx, dy in offsets], np.float32)
            for cx, cy in self.marker_centres_mm.values()
        ]
        return cv2.aruco.Board(obj_points, ARUCO_DICT, np.array(list(CORNER_NAMES)))

    def canvas_size_px(self) -> tuple[int, int]:
        return round(self.width_mm * self.px_per_mm), round(self.height_mm * self.px_per_mm)


@dataclass
class DetectedObject:
    label: str
    x_mm: float
    y_mm: float
    area_mm2: float
    confidence: float = 1.0


@dataclass
class SceneObservation:
    markers_found: list[int]
    objects: list[DetectedObject] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok}


ObjectDetector = Callable[[np.ndarray], list[DetectedObject]]


def detect_colour_objects(
    top_down_bgr: np.ndarray, px_per_mm: float, min_area_mm2: float = 150.0
) -> list[DetectedObject]:
    hsv = cv2.cvtColor(top_down_bgr, cv2.COLOR_BGR2HSV)
    found = []
    for colour, bands in COLOUR_HUES.items():
        masks = [cv2.inRange(hsv, (lo, 100, 70), (hi, 255, 255)) for lo, hi in bands]
        mask = cv2.morphologyEx(np.bitwise_or.reduce(masks), cv2.MORPH_OPEN, OPEN_KERNEL)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):  # label 0 is background
            area_mm2 = stats[i, cv2.CC_STAT_AREA] / px_per_mm**2
            if area_mm2 >= min_area_mm2:
                x, y = centroids[i] / px_per_mm
                found.append(DetectedObject(colour, round(x, 1), round(y, 1), round(area_mm2, 1)))
    return found


class ScenePerceiver:
    """Frame -> mat-frame objects: marker homography, top-down warp, pluggable detector."""

    def __init__(self, spec: MatSpec | None = None, detector: ObjectDetector | None = None) -> None:
        self.spec = spec or MatSpec()
        self.detector = detector or partial(detect_colour_objects, px_per_mm=self.spec.px_per_mm)
        self._aruco = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def observe(self, frame_bgr: np.ndarray) -> SceneObservation:
        corners, ids, _ = self._aruco.detectMarkers(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        found = sorted(int(i) for i in ids.flatten() if int(i) in CORNER_NAMES) if ids is not None else []
        if len(found) < MIN_MARKERS:
            return SceneObservation(found, error=f"need {MIN_MARKERS} corner markers, saw {len(found)}")
        obj_points, img_points = self.spec.board.matchImagePoints(corners, ids)
        homography, _ = cv2.findHomography(
            img_points.reshape(-1, 2), obj_points.reshape(-1, 3)[:, :2] * self.spec.px_per_mm
        )
        top_down = cv2.warpPerspective(frame_bgr, homography, self.spec.canvas_size_px())
        return SceneObservation(found, self.detector(top_down))
