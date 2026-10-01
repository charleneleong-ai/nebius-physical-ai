import os
from dataclasses import replace

import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import start_pose
from nebius_physical_ai.twin.scene import SO101_XML, BlockPhysics, Twin, TwinLayout, TwinRenderer

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


def test_variant_matches_a_fresh_compile() -> None:
    moved = [replace(o, x_mm=o.x_mm + 7, y_mm=o.y_mm - 4, area_mm2=(np.sqrt(o.area_mm2) + 2) ** 2) for o in OBJECTS]
    physics, pose = BlockPhysics(mass_kg=0.08, friction=0.55), start_pose(np.random.default_rng(1))
    fresh = Twin.from_objects(moved, A3, physics, pose)
    edited = Twin.from_objects(OBJECTS, A3).variant(moved, A3, physics, pose)
    for obj in moved:
        name = obj.label
        for field in ("mass", "inertia"):
            np.testing.assert_allclose(getattr(edited.model.body(name), field), getattr(fresh.model.body(name), field))
        for field in ("size", "friction", "rbound"):
            np.testing.assert_allclose(getattr(edited.model.geom(name), field), getattr(fresh.model.geom(name), field))
    fresh.settle(1.0)
    edited.settle(1.0)
    for name in edited.blocks:
        np.testing.assert_allclose(edited.block_pos(name), fresh.block_pos(name), atol=1e-4)
    np.testing.assert_allclose(edited.arm_qpos(), fresh.arm_qpos(), atol=1e-4)


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl", reason="needs headless GL")
def test_shared_renderer_matches_own_renderer_for_resized_variant() -> None:
    template = Twin.from_objects(OBJECTS, A3)
    big = [replace(o, area_mm2=(np.sqrt(o.area_mm2) + 15) ** 2) for o in OBJECTS]
    variant = template.variant(big, A3, BlockPhysics(), start_pose(np.random.default_rng(0)))
    shared = TwinRenderer(template.model, (120, 160))
    try:
        own, via_shared = variant.render("front_cam", (120, 160)), variant.render("front_cam", (120, 160), shared)
        np.testing.assert_allclose(via_shared, own, atol=4)  # GPU shading jitters by a level on a few pixels
        assert not np.array_equal(template.render("front_cam", (120, 160)), variant.render("front_cam", (120, 160)))
    finally:
        shared.close()
