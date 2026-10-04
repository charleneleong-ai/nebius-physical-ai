from dataclasses import dataclass, replace

import cv2
import numpy as np

from nebius_physical_ai.capture.camera import MarkerCalibrator
from nebius_physical_ai.capture.mat import DetectedObject, MatSpec, oriented_rect


@dataclass(frozen=True)
class MatView:
    """One calibrated view of the mat: pinhole intrinsics and the board -> camera transform.

    Board frame = mat millimetres, x right, y down the mat, z into the table (OpenCV), so height above the mat is -z.
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    size: tuple[int, int]  # (height, width)

    @classmethod
    def solve(cls, image: np.ndarray, spec: MatSpec, hfov_deg: float) -> "MatView | None":
        """Pose from the corner markers in `image`; None if fewer than two are visible."""
        calibrator = MarkerCalibrator(spec)
        corners, ids, _ = calibrator.detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < 2:
            return None
        board, pixels = spec.board.matchImagePoints(corners, ids)
        K = calibrator.intrinsics_for(hfov_deg, image.shape[:2])
        _, rvec, tvec = cv2.solvePnP(board, pixels, K, None)
        return cls(K, cv2.Rodrigues(rvec)[0], tvec.ravel(), image.shape[:2])

    @property
    def camera_height_mm(self) -> float:
        return float(-(self.R.T @ -self.t)[2])

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and straight down (board +z)."""
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1, 1))))

    def plane_hits(self) -> tuple[np.ndarray, np.ndarray]:
        """Per pixel: camera-frame depth (z) where its ray meets the mat, and that point in board coordinates."""
        h, w = self.size
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones((h, w))], -1)
        depth = (self.R[:, 2] @ self.t) / (rays @ self.R[:, 2])
        return depth, (rays * depth[..., None] - self.t) @ self.R

    def to_board(self, points: np.ndarray) -> np.ndarray:
        return (points - self.t) @ self.R


TOP_BAND = 0.8  # points above this fraction of an object's height form its top surface
ROUND_FILL = 0.88  # top-surface fill of its rectangle: disc pi/4 = 0.79, box 1.0; the midpoint splits them


class HeightLifter:
    """Heights and true footprints of perceived objects from metric point maps (e.g. MoGe-3) of calibrated views.

    A predicted point map's scale is unreliable (MoGe-3 was 19% off on the A3 mat), so each map is rescaled until the
    bare mat lands on the plane the markers define. Per object, height is a high percentile of the points standing
    above the mat (median over views) and the footprint is the rectangle around those points, which removes the
    parallax that inflates footprints measured in the mat plane; both are medians over views.
    """

    def __init__(
        self, percentile: float = 99.0, standing_mm: float = 8.0, margin_mm: float = 15.0, clearance_mm: float = 40.0
    ) -> None:
        self.percentile, self.standing_mm = percentile, standing_mm
        self.margin_mm, self.clearance_mm = margin_mm, clearance_mm

    @staticmethod
    def reach(obj: DetectedObject) -> float:
        return max(obj.length_mm, obj.width_mm, float(np.sqrt(obj.area_mm2))) / 2

    def align(
        self, view: MatView, points: np.ndarray, valid: np.ndarray, objects: list[DetectedObject], spec: MatSpec
    ) -> np.ndarray:
        """Board coordinates of the point map after fitting its scale to the bare mat."""
        depth, hit = view.plane_hits()
        x, y = hit[..., 0], hit[..., 1]
        m = spec.marker_mm / 2 + 5
        bare = valid & (depth > 0) & (x > m) & (x < spec.width_mm - m) & (y > m) & (y < spec.height_mm - m)
        for o in objects:
            bare &= np.hypot(x - o.x_mm, y - o.y_mm) > self.reach(o) + self.clearance_mm
        if bare.sum() < 100:
            raise ValueError("too little bare mat in view to fix the point map's scale")
        scale = np.median(depth[bare] / points[..., 2][bare])
        return view.to_board(points * scale)

    def lift(
        self,
        objects: list[DetectedObject],
        views: list[tuple[MatView, np.ndarray, np.ndarray]],
        spec: MatSpec,
    ) -> list[DetectedObject]:
        """`views` holds (view, point map (H, W, 3) in camera mm, valid mask). Objects nothing stands on are unchanged."""
        heights: list[list[float]] = [[] for _ in objects]
        standing: list[list[np.ndarray]] = [[] for _ in objects]
        fills: list[list[float]] = [[] for _ in objects]
        for view, points, valid in views:
            board = self.align(view, points, valid, objects, spec)
            height = -board[..., 2]
            owner, distance = self.nearest(board, objects)
            for k, o in enumerate(objects):
                near = (owner == k) & (distance < self.reach(o) + self.margin_mm)
                up = valid & near & (height > self.standing_mm)
                if up.sum() > 50:
                    top = float(np.percentile(height[up], self.percentile))
                    heights[k].append(top)
                    standing[k].append(board[up][:, :2])
                    if (lid := up & (height > TOP_BAND * top)).sum() >= 20:  # too few points: no shape vote
                        fills[k].append(self.fill(board[lid][:, :2]))
        return [self.lifted(o, *per_view) for o, *per_view in zip(objects, heights, standing, fills)]

    @staticmethod
    def nearest(board: np.ndarray, objects: list[DetectedObject]) -> tuple[np.ndarray, np.ndarray]:
        """Per pixel: index of the closest object centre and the distance to it, so neighbours can't leak in."""
        centres = np.array([[o.x_mm, o.y_mm] for o in objects])
        distances = np.linalg.norm(board[..., None, :2] - centres, axis=-1)
        return distances.argmin(-1), distances.min(-1)

    @staticmethod
    def fill(top: np.ndarray) -> float:
        """How much of its bounding rectangle an object's top surface covers: a disc pi/4, a box's lid 1."""
        hull = cv2.convexHull(top.astype(np.float32))
        (_, (a, b), _) = cv2.minAreaRect(hull)
        return float(cv2.contourArea(hull) / max(a * b, 1e-9))

    @staticmethod
    def robust_rect(points: np.ndarray, trim: float = 1.0) -> tuple[float, float, float, float, float]:
        """(x, y, length, width, yaw degrees in [-90, 90)) of the rectangle around the points, ignoring strays.

        Points beyond the `trim` percentiles along either principal axis (depth fly-points at edges) are dropped, then
        the minimum-area rectangle is fitted to the rest: it follows the outline, unlike the principal axes, which
        tilt towards whichever side faces a view happened to sample densely.
        """
        centre = np.median(points, axis=0)
        _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
        local = (points - centre) @ axes.T
        lo, hi = np.percentile(local, [trim, 100 - trim], axis=0)
        return oriented_rect(points[np.all((local >= lo) & (local <= hi), axis=1)])

    def lifted(
        self, obj: DetectedObject, heights: list[float], standing: list[np.ndarray], fills: list[float]
    ) -> DetectedObject:
        """Median over views of each view's own rectangle: a view's depth error shifts its points along its rays, so
        pooling oblique views from different directions would add their errors up as extra width."""
        if not heights:
            return obj
        rects = np.array([self.robust_rect(points) for points in standing])
        x, y, length, width = np.median(rects[:, :4], axis=0)
        doubled = np.radians(2 * rects[:, 4])  # yaw is defined modulo 180 degrees
        yaw = float(np.degrees(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2)
        round_top = bool(fills) and np.median(fills) < ROUND_FILL and length < 1.25 * width
        return replace(
            obj,
            x_mm=round(float(x), 1),
            y_mm=round(float(y), 1),
            length_mm=round(float(length), 1),
            width_mm=round(float(width), 1),
            yaw_deg=round(yaw, 1),
            height_mm=round(float(np.median(heights)), 1),
            shape="cylinder" if round_top else "box",
        )
