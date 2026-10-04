import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from nebius_physical_ai.capture.camera import CameraPose, MarkerCalibrator
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.scene import A3_MAT, SO101_XML, Twin, TwinLayout

SIZE = (480, 640)  # (h, w): 4:3 so the twin renders exactly the synthetic camera's view
CAL = MarkerCalibrator(A3_MAT)
BOARD_POINTS = np.concatenate(A3_MAT.board.getObjPoints()).astype(np.float32)  # mat frame, z into table


def camera(position_mm: tuple[float, float, float], look_at_mm: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (rvec, tvec) for a camera at a z-up mat position looking at a table point, image-up towards mat top."""
    cam = np.array([position_mm[0], position_mm[1], -position_mm[2]])  # board frame has z into the table
    z = np.array([*look_at_mm, 0.0]) - cam
    z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z)  # image right ≈ mat +x
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(z, x), z])  # rows: camera axes in the board frame
    rvec, _ = cv2.Rodrigues(rot)
    return rvec, -rot @ cam


def project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, hfov: float) -> np.ndarray:
    k = CAL.intrinsics_for(hfov, SIZE)
    return cv2.projectPoints(points, rvec, tvec, k, None)[0].reshape(-1, 2).astype(np.float32)


VIEW = ((175.0, 250.0, 420.0), (175.0, 113.0))  # like the recording: over the mat's bottom edge, ~42 cm up


def test_pose_recovers_a_known_camera() -> None:
    rvec, tvec = camera(*VIEW)
    img = project(BOARD_POINTS, rvec, tvec, 69.0)
    pose = CAL.pose_from([BOARD_POINTS], [img], SIZE, 69.0)
    assert pose.position_mm == pytest.approx(VIEW[0], abs=1.0)
    look = np.array([*VIEW[1], 0.0]) - np.array(VIEW[0])
    assert pose.tilt_deg == pytest.approx(np.degrees(np.arccos(-look[2] / np.linalg.norm(look))), abs=0.2)
    assert pose.reprojection_px < 0.01


def test_fit_hfov_from_several_viewpoints() -> None:
    views = [
        ((175 + dx, 250 + dy, 380 + dz), (175, 113))
        for dx, dy, dz in [(0, 0, 0), (-80, 20, 60), (90, -40, 30), (20, 60, -40)]
    ]
    imgs = [project(BOARD_POINTS, *camera(*v), 66.0) for v in views]
    hfov, rms = CAL.fit_hfov_from([BOARD_POINTS] * len(views), imgs, SIZE)
    assert hfov == pytest.approx(66.0, abs=0.3) and rms < 0.05


def test_pose_save_load_round_trip_restores_tuples(tmp_path: Path) -> None:
    pose = CameraPose((1.0, 2.0, 3.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), 69.0, SIZE, 0.5)
    loaded = CameraPose.load(pose.save(tmp_path / "cam.json"))
    assert loaded == pose and isinstance(loaded.image_size, tuple)  # JSON gives lists; frozen equality needs tuples


def test_crop_to_4_3_keeps_vertical_fov_and_matches_frame_crop() -> None:
    wide = CameraPose((0.0, 0.0, 400.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), 69.0, (720, 1280), 0.0)
    cropped = wide.cropped_to(4 / 3)
    assert cropped.image_size == (720, 960) and cropped.vfov_deg == pytest.approx(wide.vfov_deg)
    assert cropped.hfov_deg < wide.hfov_deg
    assert wide.crop_frame(np.zeros((720, 1280, 3)), 4 / 3).shape[:2] == (720, 960)


@pytest.mark.skipif(not SO101_XML.exists() or os.environ.get("MUJOCO_GL") != "egl", reason="needs model + headless GL")
def test_twin_camera_sees_the_cube_where_opencv_projects_it() -> None:
    rvec, tvec = camera(*VIEW)
    pose = CAL.pose_from([BOARD_POINTS], [project(BOARD_POINTS, rvec, tvec, 69.0)], SIZE, 69.0)
    cube = DetectedObject("red", 250.0, 60.0, 30.0**2)  # 30 mm cube, away from the arm
    twin = Twin.from_objects([cube], TwinLayout(front_camera=pose))
    twin.settle(0.2)
    img = twin.render("front_cam", SIZE)
    red = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
    seen = np.array(np.nonzero(red)[::-1]).mean(axis=1)  # (u, v)
    top_face_centre = np.array([[cube.x_mm, cube.y_mm, -30.0]], dtype=np.float32)  # board frame: z into table
    expected = project(top_face_centre, rvec, tvec, 69.0)[0]
    assert np.linalg.norm(seen - expected) < 6  # pixels; the visible blob includes a sliver of the side faces
