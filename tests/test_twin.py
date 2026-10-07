import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import SceneSampler, start_pose
from nebius_physical_ai.twin.randomize import SceneRandomizer
from nebius_physical_ai.twin.rollout import FailureWatch, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, BlockPhysics, Twin, TwinLayout, TwinRenderer

pytestmark = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")

A3 = TwinLayout()
OBJECTS = [
    DetectedObject("red", 175.0, 113.0, 900.0),
    DetectedObject("blue", 60.0, 40.0, 1600.0),
    DetectedObject("green", 300.0, 200.0, 400.0),
    DetectedObject(
        "yellow", 60.0, 120.0, 9600.0, length_mm=160.0, width_mm=60.0, yaw_deg=30.0, height_mm=30.0
    ),  # glasses case
    DetectedObject(
        "yellow",
        300.0,
        60.0,
        3800.0,
        length_mm=70.0,
        width_mm=70.0,
        height_mm=40.0,
        shape="cylinder",
        rgb=(0.78, 0.7, 0.48),
    ),  # pomade tin, measured lid colour
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
        assert pos[:2] == pytest.approx(A3.mat_to_world(obj.x_mm, obj.y_mm), abs=0.003)
        assert pos[2] == pytest.approx(twin.geometry(name).half[2], abs=0.002)


def test_repeated_colours_get_suffixed_names() -> None:
    assert Twin.block_names(OBJECTS) == ["red", "blue", "green", "yellow_0", "yellow_1"]


def test_box_and_cylinder_follow_the_perceived_footprint() -> None:
    twin = Twin.from_objects(OBJECTS, A3)
    case, tin = twin.model.geom("yellow_0"), twin.model.geom("yellow_1")
    assert case.size == pytest.approx([0.08, 0.03, 0.015]) and tin.size[:2] == pytest.approx([0.035, 0.02])
    mujoco.mj_forward(twin.model, twin.data)
    long_axis = twin.data.body("yellow_0").xmat.reshape(3, 3)[:, 0]
    theta = np.radians(30.0)
    expected = TwinLayout.direction_to_world((np.cos(theta), np.sin(theta), 0.0))
    assert abs(long_axis @ expected) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl", reason="needs headless GL")
@pytest.mark.parametrize("camera", ["front_cam", "wrist_cam"])
def test_cameras_render(camera: str) -> None:
    twin = Twin.from_objects(OBJECTS, A3)
    twin.settle(0.2)
    img = twin.render(camera)
    assert img.shape == (480, 640, 3) and img.std() > 10


def test_variant_matches_a_fresh_compile() -> None:
    moved = [
        replace(
            o,
            x_mm=o.x_mm + 7,
            y_mm=o.y_mm - 4,
            area_mm2=o.area_mm2 * 1.1,
            length_mm=o.length_mm * 1.05,
            width_mm=o.width_mm * 1.05,
            yaw_deg=o.yaw_deg + 12,
        )
        for o in OBJECTS
    ]
    physics, pose = BlockPhysics(mass_kg=0.08, friction=0.55), start_pose(np.random.default_rng(1))
    fresh = Twin.from_objects(moved, A3, physics, pose)
    edited = Twin.from_objects(OBJECTS, A3).variant(moved, A3, physics, pose)
    for name in Twin.block_names(moved):
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


def test_measured_colour_paints_the_block() -> None:
    twin = Twin.from_objects(OBJECTS, A3)
    assert twin.model.geom("yellow_1").rgba == pytest.approx([0.78 * 0.6, 0.7 * 0.6, 0.48 * 0.6, 1.0])  # paper -> floor
    assert twin.model.geom("yellow_0").rgba[:3] == pytest.approx([0.95, 0.8, 0.1])  # no measurement: palette


def test_unknown_block_names_the_real_ones() -> None:
    with pytest.raises(KeyError, match="yellow_0"):
        Twin.from_objects(OBJECTS, A3).block_pos("yellow")


@pytest.mark.parametrize(("along_mm", "stacked"), [(50.0, True), (100.0, False)])
def test_stack_check_uses_the_bottom_blocks_long_axis(along_mm: float, stacked: bool) -> None:
    long_red = DetectedObject("red", 175.0, 113.0, 9600.0, length_mm=160.0, width_mm=60.0, height_mm=30.0)
    twin = Twin.from_objects([DetectedObject("blue", 60.0, 40.0, 900.0), long_red], A3)
    adr = twin.model.jnt_qposadr[twin.model.body("blue").jntadr[0]]
    long_axis = TwinLayout.direction_to_world((1.0, 0.0, 0.0))  # yaw 0: the footprint's long side runs along mat +x
    twin.data.qpos[adr : adr + 3] = twin.block_pos("red") + along_mm / 1000 * long_axis + [0, 0, 0.05]
    twin.settle(1.0)
    assert StackCheck()(twin) is stacked


def test_front_camera_keeps_the_photos_orientation() -> None:
    """Calibrated pose (camera over the mat's bottom edge, image up = mat top): the axis hits the image centre, mat +x
    goes right and mat -y goes up, so the twin's view isn't mirrored or flipped."""
    pose = CameraPose((176.0, 236.0, 420.0), (0.005, -0.24, -0.971), (0.004, -0.971, 0.24), 68.8, (720, 1280), 0.0)
    layout = TwinLayout(front_camera=pose)
    twin = Twin.from_objects([OBJECTS[0]], layout)
    position, forward = np.array(pose.position_mm), np.array(pose.forward)
    hit = position - position[2] / forward[2] * forward  # where the optical axis meets the mat
    mat_points = [hit[:2], hit[:2] + [30, 0], hit[:2] + [0, -30]]
    world = np.array([[*layout.mat_to_world(x, y), 0.0] for x, y in mat_points])
    (u0, v0), (u_right, v_right), (u_up, v_up) = twin.project(world, "front_cam")
    assert (u0, v0) == pytest.approx((320, 240), abs=1.0)
    assert u_right > u0 + 20 and abs(v_right - v0) < 5
    assert v_up < v0 - 20 and abs(u_up - u0) < 5


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl", reason="needs headless GL")
def test_randomized_scene_compiles_and_shows_the_task_blocks() -> None:
    sampler, randomizer = SceneSampler(A3, seed=4), SceneRandomizer(A3, seed=4)
    look = randomizer.look()
    with Twin.from_objects(randomizer.objects(sampler), A3, start_pose=sampler.start_pose(), look=look) as twin:
        assert twin.model.cam("front_cam").pos == pytest.approx(look.camera["pos"])
        image = twin.render("front_cam", (240, 320))
        assert image.std() > 10  # a real scene, not a blank frame
        twin.settle(0.5)
        assert all(twin.block_pos(name)[2] > 0 for name in twin.blocks)  # nothing knocked through the floor


class TestFailureWatch:
    """Each failure is staged by moving the blocks the way a rollout would, then asking for the verdict."""

    @staticmethod
    def move(twin: Twin, name: str, offset: tuple[float, float, float]) -> None:
        adr = twin.model.jnt_qposadr[twin.model.body(name).jntadr[0]]
        twin.data.qpos[adr : adr + 3] += offset
        mujoco.mj_forward(twin.model, twin.data)

    @pytest.fixture
    def twin(self) -> Twin:
        return Twin.from_objects(
            [DetectedObject("blue", 131.0, 151.0, 900.0), DetectedObject("red", 221.0, 151.0, 1600.0)], A3
        )

    def stage(self, twin: Twin, path: list[tuple[str, tuple[float, float, float]]]) -> str:
        watch = FailureWatch(twin, StackCheck())
        for name, offset in path:
            self.move(twin, name, offset)
            watch.update(twin)
        return watch.verdict(twin)

    def test_untouched_blocks_mean_no_contact(self, twin: Twin) -> None:
        assert self.stage(twin, []) == "no_contact"

    def test_sliding_the_pick_block_is_a_push(self, twin: Twin) -> None:
        assert self.stage(twin, [("blue", (0.03, 0.0, 0.0))]) == "pushed"

    def test_lifting_then_setting_down_away_from_the_target_is_a_drop(self, twin: Twin) -> None:
        assert self.stage(twin, [("blue", (0.0, 0.0, 0.05)), ("blue", (0.03, 0.0, -0.05))]) == "dropped"

    def test_carried_over_the_target_but_not_on_it_fell_off(self, twin: Twin) -> None:
        red, blue = twin.block_pos("red"), twin.block_pos("blue")
        over = (red[0] - blue[0], red[1] - blue[1], 0.06)
        assert self.stage(twin, [("blue", over), ("blue", (0.05, 0.0, -0.06))]) == "fell_off"

    def test_still_in_the_air_at_the_end_is_holding(self, twin: Twin) -> None:
        assert self.stage(twin, [("blue", (0.0, 0.0, 0.05))]) == "holding"

    def test_shoving_the_place_block_first_is_knocked_place(self, twin: Twin) -> None:
        assert self.stage(twin, [("red", (0.03, 0.0, 0.0))]) == "knocked_place"
