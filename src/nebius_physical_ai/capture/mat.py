from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
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
    "purple": [(131, 160)],
}
ROUND_CIRCULARITY = 0.8  # 4πA/P²: 1 for a disc, π/4 for a square
OPEN_KERNEL = np.ones((5, 5), np.uint8)
# The mat is near-white (S~15, V~200): anything more saturated or darker is an object, pale lids and black tins included.
FOREGROUND_SAT, FOREGROUND_DARK = 60, 80


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
    """An object on the mat. Footprint fields are 0 when unknown (then the twin assumes a cube from the area)."""

    label: str
    x_mm: float
    y_mm: float
    area_mm2: float
    confidence: float = 1.0
    length_mm: float = 0.0  # long side of the footprint
    width_mm: float = 0.0
    yaw_deg: float = 0.0  # long side's angle from mat +x towards mat +y (down the mat)
    height_mm: float = 0.0
    shape: str = "box"  # "box" or "cylinder"
    # Median colour relative to the mat's white paper (paper = 1), so exposure and white balance cancel; None = palette.
    rgb: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.rgb is not None:
            self.rgb = tuple(self.rgb)  # JSON brings it back as a list


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


class MarkersNotFound(ValueError):
    """Fewer than MIN_MARKERS corner markers were visible, so the mat can't be located in the frame."""

    def __init__(self, found: list[int]) -> None:
        super().__init__(f"need {MIN_MARKERS} corner markers, saw {len(found)}")
        self.found = found


@dataclass(frozen=True)
class MatLocation:
    """Where the mat sits in a frame, from its corner markers.

    `homography` maps frame pixels to mat pixels (`px_per_mm` per millimetre), the warp behind the top-down view.
    `residual_mm` is the RMS distance between the markers' printed corners and where that homography puts the
    detected ones: large means a bent sheet, a wrong MatSpec or a bad detection.
    """

    markers_found: list[int]
    homography: np.ndarray
    px_per_mm: float
    residual_mm: float

    def to_mat_mm(self, points_px: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
        """Frame pixels (N, 2) on the mat plane to mat millimetres (N, 2)."""
        pts = np.asarray(points_px, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2) / self.px_per_mm


def detect_colour_objects(
    top_down_bgr: np.ndarray,
    px_per_mm: float,
    min_area_mm2: float = 150.0,
    corner_mm: float = 0.0,
    merge_mm: float = 6.0,
) -> list[DetectedObject]:
    """Blobs that stand out from the light mat, labelled by their dominant colour (or "dark").

    Touching parts within `merge_mm` count as one object (a two-tone tin, a cluster of balls); the
    `corner_mm` squares at each corner hold marker quarters and are ignored.
    """
    hsv = cv2.cvtColor(top_down_bgr, cv2.COLOR_BGR2HSV)
    colours = {
        colour: np.bitwise_or.reduce([cv2.inRange(hsv, (lo, 100, 70), (hi, 255, 255)) for lo, hi in bands])
        for colour, bands in COLOUR_HUES.items()
    }
    foreground = cv2.inRange(hsv, (0, FOREGROUND_SAT, 0), (179, 255, 255)) | cv2.inRange(
        hsv, (0, 0, 0), (179, 255, FOREGROUND_DARK)
    )
    mask = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, OPEN_KERNEL)
    markers = np.zeros(mask.shape, bool)
    if (c := round(corner_mm * px_per_mm)) > 0:
        markers[:c, :c] = markers[:c, -c:] = markers[-c:, :c] = markers[-c:, -c:] = True
        mask[markers] = 0
    k = max(1, round(merge_mm * px_per_mm))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    paper = np.median(top_down_bgr[(mask == 0) & ~markers], axis=0)
    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area_mm2 = cv2.contourArea(contour) / px_per_mm**2
        if area_mm2 >= min_area_mm2:
            blob = cv2.drawContours(np.zeros_like(mask), [contour], -1, 255, -1)
            counts = {colour: cv2.countNonZero(m & blob) for colour, m in colours.items()}
            label = max(counts, key=counts.__getitem__) if any(counts.values()) else "dark"
            b, g, r = np.clip(np.median(top_down_bgr[blob > 0], axis=0) / np.maximum(paper, 1), 0, 1)
            found.append(footprint(label, contour, px_per_mm, area_mm2, (round(r, 3), round(g, 3), round(b, 3))))
    return found


def oriented_rect(points: np.ndarray) -> tuple[float, float, float, float, float]:
    """(centre x, centre y, length, width, yaw degrees in [-90, 90)) of the smallest rectangle around 2D points."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    corners = cv2.boxPoints(rect)
    short_edge, long_edge = sorted([corners[1] - corners[0], corners[2] - corners[1]], key=np.linalg.norm)
    yaw = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
    yaw = (yaw + 90.0) % 180.0 - 90.0  # a footprint is symmetric under 180°
    return *(float(c) for c in rect[0]), float(np.linalg.norm(long_edge)), float(np.linalg.norm(short_edge)), yaw


def footprint(
    label: str, contour: np.ndarray, px_per_mm: float, area_mm2: float, rgb: tuple[float, float, float] | None = None
) -> DetectedObject:
    """Oriented rectangle around a blob (or a disc if it's round), in mat millimetres."""
    x, y, length, width, yaw = oriented_rect(contour)
    x, y, length, width = (v / px_per_mm for v in (x, y, length, width))
    circularity = 4 * np.pi * cv2.contourArea(contour) / max(cv2.arcLength(contour, True) ** 2, 1e-9)
    shape = "cylinder" if circularity > ROUND_CIRCULARITY and length < 1.25 * width else "box"
    return DetectedObject(
        label,
        round(x, 1),
        round(y, 1),
        round(area_mm2, 1),
        length_mm=round(length, 1),
        width_mm=round(width, 1),
        yaw_deg=round(yaw, 1),
        shape=shape,
        rgb=rgb,
    )


class ScenePerceiver:
    """Frame -> mat-frame objects: marker homography, top-down warp, pluggable detector."""

    def __init__(self, spec: MatSpec | None = None, detector: ObjectDetector | None = None) -> None:
        self.spec = spec or MatSpec()
        self.detector = detector or partial(
            detect_colour_objects, px_per_mm=self.spec.px_per_mm, corner_mm=self.spec.marker_mm / 2
        )
        self._aruco = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def locate(self, frame_bgr: np.ndarray) -> MatLocation:
        """The mat's pixel -> millimetre homography and how well the marker corners fit it.

        Raises MarkersNotFound when fewer than MIN_MARKERS corner markers are visible.
        """
        corners, ids, _ = self._aruco.detectMarkers(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        found = sorted(int(i) for i in ids.flatten() if int(i) in CORNER_NAMES) if ids is not None else []
        if len(found) < MIN_MARKERS:
            raise MarkersNotFound(found)
        obj_points, img_points = self.spec.board.matchImagePoints(corners, ids)
        printed_mm = obj_points.reshape(-1, 3)[:, :2]
        homography, _ = cv2.findHomography(img_points.reshape(-1, 2), printed_mm * self.spec.px_per_mm)
        location = MatLocation(found, homography, self.spec.px_per_mm, 0.0)
        fitted_mm = location.to_mat_mm(img_points.reshape(-1, 2))
        residual = float(np.sqrt(np.mean(np.sum((fitted_mm - printed_mm) ** 2, axis=1))))
        return replace(location, residual_mm=residual)

    def perceive(self, frame_bgr: np.ndarray) -> tuple[SceneObservation, MatLocation | None]:
        """`observe`, plus the marker location it was built on (None when the mat couldn't be located)."""
        try:
            location = self.locate(frame_bgr)
        except MarkersNotFound as err:
            return SceneObservation(err.found, error=str(err)), None
        top_down = cv2.warpPerspective(frame_bgr, location.homography, self.spec.canvas_size_px())
        return SceneObservation(location.markers_found, self.detector(top_down)), location

    def observe(self, frame_bgr: np.ndarray) -> SceneObservation:
        return self.perceive(frame_bgr)[0]
