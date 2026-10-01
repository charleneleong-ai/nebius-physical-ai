import os

import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout

pytestmark = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")

A3 = TwinLayout()
OBJECTS = [
    DetectedObject("red", 175.0, 113.0, 900.0),
    DetectedObject("blue", 60.0, 40.0, 1600.0),
    DetectedObject("green", 300.0, 200.0, 400.0),
]


@pytest.mark.parametrize(
    ("mat_xy", "world_xy"),
    [((175, 277), (0.0, 0.0)), ((175, 0), (0.277, 0.0)), ((0, 227), (0.05, 0.175))],
)
def test_mat_to_world(mat_xy: tuple[float, float], world_xy: tuple[float, float]) -> None:
    assert A3.mat_to_world(*mat_xy) == pytest.approx(world_xy)


def test_blocks_settle_where_perceived() -> None:
    twin = Twin.from_objects(OBJECTS, A3)
    assert twin.model.nu == 6
    twin.settle(1.0)
    for name, obj in twin.blocks.items():
        pos = twin.block_pos(name)
        side = Twin.block_side_m(obj)
        assert pos[:2] == pytest.approx(A3.mat_to_world(obj.x_mm, obj.y_mm), abs=0.003)
        assert pos[2] == pytest.approx(side / 2, abs=0.002)


def test_duplicate_labels_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Twin.from_objects([OBJECTS[0], OBJECTS[0]], A3)


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl", reason="needs headless GL")
@pytest.mark.parametrize("camera", ["front_cam", "wrist_cam"])
def test_cameras_render(camera: str) -> None:
    twin = Twin.from_objects(OBJECTS, A3)
    twin.settle(0.2)
    img = twin.render(camera)
    assert img.shape == (480, 640, 3) and img.std() > 10
