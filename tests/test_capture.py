import json
from dataclasses import asdict

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nebius_physical_ai.capture.markers import render_marker_sheet
from nebius_physical_ai.capture.mat import ARUCO_DICT, DetectedObject, MatSpec, ScenePerceiver, detect_colour_objects
from nebius_physical_ai.capture.server import create_app

SPEC = MatSpec()
MARGIN_MM = 40
PX = 3  # synthetic scene resolution, px per mm
OBJECTS = {"red": (100.0, 150.0), "blue": (220.0, 80.0), "green": (60.0, 240.0)}
BGR = {"red": (0, 0, 220), "blue": (220, 60, 0), "green": (0, 200, 0)}


def mm(v: float) -> int:
    return round((v + MARGIN_MM) * PX)


def top_down_scene(markers: tuple[int, ...] = (0, 1, 2, 3)) -> np.ndarray:
    size = mm(SPEC.width_mm + MARGIN_MM), mm(SPEC.height_mm + MARGIN_MM)
    img = np.full((size[1], size[0], 3), 235, np.uint8)
    half = round(SPEC.marker_mm * PX / 2)
    for i, (x, y) in SPEC.marker_centres_mm.items():
        if i in markers:
            marker = cv2.aruco.generateImageMarker(ARUCO_DICT, i, 2 * half)
            img[mm(y) - half : mm(y) + half, mm(x) - half : mm(x) + half] = marker[..., None]
    for colour, (x, y) in OBJECTS.items():
        cv2.rectangle(img, (mm(x - 15), mm(y - 15)), (mm(x + 15), mm(y + 15)), BGR[colour], -1)
    return img


def phone_view(img: np.ndarray, skew: float) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * skew, h * 0.05], [w * (1 - skew), 0], [w, h], [0, h * 0.95]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    return cv2.GaussianBlur(out, (3, 3), 0)


@pytest.mark.parametrize(
    ("skew", "visible"),
    [(0.0, (0, 1, 2, 3)), (0.12, (0, 1, 2, 3)), (0.2, (0, 1, 2, 3)), (0.12, (0, 2))],
)
def test_recovers_object_positions_under_perspective(skew: float, visible: tuple[int, ...]) -> None:
    obs = ScenePerceiver(SPEC).observe(phone_view(top_down_scene(visible), skew))
    assert obs.ok, obs.error
    assert obs.markers_found == list(visible)
    got = {o.label: (o.x_mm, o.y_mm) for o in obs.objects}
    assert got.keys() == OBJECTS.keys()
    for colour, (x, y) in OBJECTS.items():
        assert got[colour] == pytest.approx((x, y), abs=4.0)


def test_reports_too_few_markers() -> None:
    obs = ScenePerceiver(SPEC).observe(top_down_scene(markers=(1,)))
    assert not obs.ok and obs.objects == [] and "saw 1" in obs.error


def test_marker_sheet_markers_are_detectable() -> None:
    sheet = cv2.cvtColor(np.asarray(render_marker_sheet(SPEC).convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, ids, _ = cv2.aruco.ArucoDetector(ARUCO_DICT).detectMarkers(sheet)
    assert sorted(ids.flatten().tolist()) == [0, 1, 2, 3]


def test_frame_endpoint_round_trip() -> None:
    client = TestClient(create_app())
    _, jpeg = cv2.imencode(".jpg", phone_view(top_down_scene(), 0.1))
    body = client.post("/frame", content=jpeg.tobytes()).json()
    assert body["ok"]
    assert client.get("/latest").json() == body
    assert client.post("/frame", content=b"not an image").status_code == 400
    assert "getUserMedia" in client.get("/").text


@pytest.mark.parametrize("yaw_deg", [-60.0, 0.0, 30.0])
def test_footprint_measures_rotated_box_and_disc(yaw_deg: float) -> None:
    img = np.full((300 * PX, 300 * PX, 3), 235, np.uint8)
    t = np.radians(yaw_deg)
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    corners = [np.array([100.0, 150.0]) + a * 80 * u + b * 30 * v for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    cv2.fillPoly(img, [np.int32(np.array(corners) * PX)], BGR["blue"])
    cv2.circle(img, (230 * PX, 150 * PX), 35 * PX, BGR["red"], -1)
    found = {o.label: o for o in detect_colour_objects(img, px_per_mm=PX)}
    box, disc = found["blue"], found["red"]
    assert box.shape == "box" and (box.length_mm, box.width_mm) == pytest.approx((160, 60), abs=2)
    assert box.yaw_deg == pytest.approx(yaw_deg, abs=1.5)
    assert box.rgb == pytest.approx((0, 60 / 235, 220 / 235), abs=0.02)  # relative to the mat's paper (235)
    assert disc.shape == "cylinder" and disc.length_mm == pytest.approx(70, abs=2)


def test_two_tone_object_is_one_detection() -> None:
    img = np.full((300 * PX, 300 * PX, 3), 235, np.uint8)
    centre = (150 * PX, 150 * PX)
    cv2.ellipse(img, centre, (40 * PX, 40 * PX), 0, 0, 180, (40, 40, 40), -1)  # black tin side
    cv2.ellipse(img, centre, (40 * PX, 40 * PX), 0, 180, 360, (60, 200, 230), -1)  # yellow lid
    (tin,) = detect_colour_objects(img, px_per_mm=PX)
    assert tin.shape == "cylinder" and tin.label == "yellow"
    assert (tin.x_mm, tin.y_mm, tin.length_mm) == pytest.approx((150, 150, 80), abs=2)


def test_detected_object_survives_a_json_round_trip() -> None:
    obj = DetectedObject("yellow", 1.0, 2.0, 3.0, rgb=(0.1, 0.2, 0.3))
    assert DetectedObject(**json.loads(json.dumps(asdict(obj)))) == obj
