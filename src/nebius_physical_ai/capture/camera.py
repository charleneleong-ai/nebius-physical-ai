import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from nebius_physical_ai.capture.mat import ARUCO_DICT, MatSpec

IPHONE_MAIN_HFOV_DEG = 69.0  # 26 mm-equivalent wide camera in 16:9 video, after stabilisation crop


@dataclass(frozen=True)
class CameraPose:
    """A camera relative to the mat: position in mat millimetres (z up from the table) and its viewing rays.

    `forward`/`up` are unit vectors in the mat frame (x right along the mat, y down the mat, z up).
    """

    position_mm: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    hfov_deg: float
    image_size: tuple[int, int]  # (height, width)
    reprojection_px: float

    @property
    def vfov_deg(self) -> float:
        h, w = self.image_size
        return float(np.degrees(2 * np.arctan(np.tan(np.radians(self.hfov_deg) / 2) * h / w)))

    def cropped_to(self, aspect: float) -> "CameraPose":
        """The same camera after centre-cropping its image to width/height = `aspect` (FOV shrinks on the cut axis)."""
        h, w = self.image_size
        new_h, new_w = (h, round(h * aspect)) if w / h > aspect else (round(w / aspect), w)
        half = np.tan(np.radians(self.hfov_deg) / 2) * new_w / w
        return replace(self, image_size=(new_h, new_w), hfov_deg=float(np.degrees(2 * np.arctan(half))))

    def crop_frame(self, frame: np.ndarray, aspect: float) -> np.ndarray:
        """Centre-crop a real frame from this camera to `aspect`, matching `cropped_to(aspect)`."""
        new_h, new_w = self.cropped_to(aspect).image_size
        h, w = frame.shape[:2]
        top, left = (h - new_h) // 2, (w - new_w) // 2
        return frame[top : top + new_h, left : left + new_w]

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and straight down."""
        return float(np.degrees(np.arccos(np.clip(-self.forward[2], -1, 1))))

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "CameraPose":
        d = json.loads(path.read_text())
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class MarkerCalibrator:
    """Camera intrinsics and pose from video of the printed mat, using the corner markers as a known 3D target.

    Mat frame for the board: x right, y down the mat, z into the table (OpenCV convention); poses are converted
    to z-up on output.
    """

    def __init__(self, spec: MatSpec) -> None:
        self.spec = spec
        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def correspondences(
        self, video: Path, every: int = 10
    ) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
        """Per sampled frame with >= 2 markers: (board points, image points); plus image size (h, w)."""
        cap = cv2.VideoCapture(str(video))
        objs, imgs, size, i = [], [], (0, 0), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            size = frame.shape[:2]
            if i % every == 0:
                corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if ids is not None and len(ids) >= 2:
                    obj, img = self.spec.board.matchImagePoints(corners, ids)
                    objs.append(obj.astype(np.float32))
                    imgs.append(img.astype(np.float32))
            i += 1
        cap.release()
        if not objs:
            raise ValueError(f"no frame in {video} shows two corner markers")
        return objs, imgs, size

    @staticmethod
    def unit(v: np.ndarray) -> tuple[float, float, float]:
        return tuple(float(x) for x in v / np.linalg.norm(v))

    @staticmethod
    def intrinsics_for(hfov_deg: float, size: tuple[int, int]) -> np.ndarray:
        h, w = size
        f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)

    def fit_hfov(self, video: Path) -> tuple[float, float]:
        """Focal length from many viewpoints (move the phone around the mat). Returns (hfov degrees, RMS px)."""
        return self.fit_hfov_from(*self.correspondences(video))

    def pose(self, video: Path, hfov_deg: float = IPHONE_MAIN_HFOV_DEG) -> CameraPose:
        """Median camera pose over a clip from a fixed mount."""
        return self.pose_from(*self.correspondences(video), hfov_deg)

    def fit_hfov_from(
        self, objs: list[np.ndarray], imgs: list[np.ndarray], size: tuple[int, int]
    ) -> tuple[float, float]:
        """Square pixels, centred principal point and no distortion: enough for a phone's main camera."""
        h, w = size
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS
        guess = self.intrinsics_for(IPHONE_MAIN_HFOV_DEG, (h, w))
        rms, k, *_ = cv2.calibrateCamera(objs, imgs, (w, h), guess, np.zeros(5), flags=flags)
        return float(np.degrees(2 * np.arctan(w / (2 * k[0, 0])))), float(rms)

    def pose_from(
        self, objs: list[np.ndarray], imgs: list[np.ndarray], size: tuple[int, int], hfov_deg: float
    ) -> CameraPose:
        k = self.intrinsics_for(hfov_deg, size)
        positions, forwards, ups, errors = [], [], [], []
        flip = np.diag([1.0, 1.0, -1.0])  # board z points into the table; output z points up
        for obj, img in zip(objs, imgs):
            ok, rvec, tvec = cv2.solvePnP(obj, img, k, None)
            if not ok:
                continue
            rot, _ = cv2.Rodrigues(rvec)
            positions.append(flip @ (-rot.T @ tvec).ravel())
            forwards.append(flip @ rot.T @ np.array([0.0, 0.0, 1.0]))  # optical axis
            ups.append(flip @ rot.T @ np.array([0.0, -1.0, 0.0]))  # image up is -y in OpenCV
            projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
            errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, 1)))))
        forward, up = np.median(forwards, axis=0), np.median(ups, axis=0)
        forward = forward / np.linalg.norm(forward)
        return CameraPose(
            position_mm=tuple(float(x) for x in np.median(positions, axis=0)),
            forward=self.unit(forward),
            up=self.unit(up - (up @ forward) * forward),  # medians drift off orthogonal
            hfov_deg=hfov_deg,
            image_size=size,
            reprojection_px=float(np.median(errors)),
        )
