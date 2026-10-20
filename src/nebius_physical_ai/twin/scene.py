import copy
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self

import mujoco
import numpy as np

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject, MatSpec
from nebius_physical_ai.policy.gr00t import CAMERA_KEYS, Observation
from nebius_physical_ai.twin.textures import SIZE, texture

SO101_XML = Path(
    os.environ.get(
        "PREFLIGHT_SO101_XML",
        Path.home() / ".cache/preflight/mujoco_menagerie/robotstudio_so101/so101.xml",
    )
)
RGBA = {
    "red": (0.85, 0.1, 0.1, 1),
    "yellow": (0.95, 0.8, 0.1, 1),
    "green": (0.1, 0.7, 0.2, 1),
    "blue": (0.1, 0.3, 0.9, 1),
    "purple": (0.5, 0.2, 0.7, 1),
    "dark": (0.12, 0.12, 0.12, 1),
}
BLOCK_SIDE_MM = (20.0, 60.0)
# The floor stands in for the mat's paper; perceived colours are paper-relative, so blocks are painted relative to it.
FLOOR_RGBA = (0.6, 0.6, 0.6, 1.0)
# Robonine SO-ARM101 follower: white printed parts, black STS3215 servos (the menagerie model ships yellow).
ARM_PRINT_RGBA = (0.93, 0.93, 0.91, 1.0)
SERVO_RGBA = (0.08, 0.08, 0.08, 1.0)
RENDER_ASPECT = 4 / 3  # GR00T's SO-101 checkpoint sees 640x480
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
A3_MAT = MatSpec(width_mm=350.0, height_mm=227.0)  # the printed A3 sheet; MatSpec defaults to the A4 cut-outs


def lerobot_to_qpos(joints: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """LeRobot units (arm joints in [-100, 100], gripper in [0, 100]) to radians across `limits` (6, 2)."""
    lo, hi = limits.T
    unit = np.concatenate([(joints[..., :5] + 100) / 200, joints[..., 5:] / 100], axis=-1)
    return lo + np.clip(unit, 0, 1) * (hi - lo)


@dataclass(frozen=True)
class TwinLayout:
    """Robot base sits beyond the mat's bottom edge, centred, facing up the mat.

    World frame = robot base frame: +x forward (towards mat top), +y left (towards mat x=0), z up.
    """

    mat: MatSpec = A3_MAT
    base_offset_mm: float = 50.0
    front_camera: CameraPose | None = None  # calibrated real camera; None = straight down from 60 cm

    def mat_to_world(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        base_x, base_y = self.mat.width_mm / 2, self.mat.height_mm + self.base_offset_mm
        return (base_y - y_mm) / 1000, (base_x - x_mm) / 1000

    @staticmethod
    def direction_to_world(v: tuple[float, float, float]) -> np.ndarray:
        """Mat-frame direction (x right, y down the mat, z up) to world (x forward, y left, z up)."""
        return np.array([-v[1], -v[0], v[2]])

    def top_down_camera_spec(self) -> dict[str, list[float] | float]:
        """Straight down from 60 cm over the mat's centre: the view the v1 and v2 checkpoints were trained on."""
        top_x, _ = self.mat_to_world(self.mat.width_mm / 2, self.mat.height_mm / 2)
        return {"pos": [top_x, 0.0, 0.6], "xyaxes": [0, -1, 0, 1, 0, 0], "fovy": 45.0}

    def camera_spec(self, pose: CameraPose) -> dict[str, list[float] | float]:
        """`add_camera` kwargs for a camera pose measured on the mat, cropped to RENDER_ASPECT: feed the policy real
        frames through `CameraPose.crop_frame(frame, RENDER_ASPECT)` for the views to match."""
        cam = pose.cropped_to(RENDER_ASPECT)
        x, y = self.mat_to_world(cam.position_mm[0], cam.position_mm[1])
        forward, up = self.direction_to_world(cam.forward), self.direction_to_world(cam.up)
        right = np.cross(forward, up)  # MuJoCo cameras look down -z with y up, so x = y × z = forward × up
        return {"pos": [x, y, cam.position_mm[2] / 1000], "xyaxes": [*right, *up], "fovy": cam.vfov_deg}

    def front_camera_spec(self) -> dict[str, list[float] | float]:
        """The layout's own camera: the calibrated one when there is one, else straight down."""
        return self.top_down_camera_spec() if self.front_camera is None else self.camera_spec(self.front_camera)


@dataclass(frozen=True)
class SceneLook:
    """What one episode looks like: top camera, key light, ambient and mat brightness. Defaults are the standard scene.

    `camera` holds `add_camera` kwargs in the world frame (None = the layout's camera).
    """

    camera: dict[str, list[float] | float] | None = None
    light_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    light_diffuse: float = 0.7
    ambient: float = 0.1
    floor_grey: float = FLOOR_RGBA[0]
    floor_texture: str = ""  # a `twin.textures` name such as "wood:3"; empty = flat grey
    # Wrist camera in the gripper's camera-mount frame ({"pos", "quat", "fovy"}); None = the menagerie mount.
    wrist_camera: dict[str, Any] | None = None


@dataclass(frozen=True)
class BlockGeometry:
    """A perceived object as a MuJoCo primitive: half extents along (long side, short side, up) and world yaw."""

    shape: str
    half: tuple[float, float, float]  # metres
    yaw_rad: float

    FOOTPRINT_MM = (10.0, 250.0)

    @classmethod
    def of(cls, obj: DetectedObject) -> "BlockGeometry":
        if obj.length_mm <= 0:  # no footprint: a cube from the blob area
            side = float(np.clip(np.sqrt(obj.area_mm2), *BLOCK_SIDE_MM)) / 1000
            return cls("box", (side / 2,) * 3, 0.0)
        length, width = (float(np.clip(v, *cls.FOOTPRINT_MM)) for v in (obj.length_mm, obj.width_mm))
        height = obj.height_mm or float(np.clip(width, *BLOCK_SIDE_MM))  # unknown height: as tall as it is narrow
        theta = np.radians(obj.yaw_deg)
        yaw = -np.pi / 2 - theta  # mat direction (cos, sin) is world (-sin, -cos)
        if obj.shape == "cylinder":
            r = (length + width) / 4000
            return cls("cylinder", (r, r, height / 2000), yaw)
        return cls("box", (length / 2000, width / 2000, height / 2000), yaw)

    @property
    def geom_type(self) -> mujoco.mjtGeom:
        return mujoco.mjtGeom.mjGEOM_CYLINDER if self.shape == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX

    @property
    def size(self) -> list[float]:
        hx, hy, hz = self.half
        return [hx, hz, 0.0] if self.shape == "cylinder" else [hx, hy, hz]

    @property
    def rbound(self) -> float:
        hx, _, hz = self.half
        return float(np.hypot(hx, hz)) if self.shape == "cylinder" else float(np.linalg.norm(self.half))

    @property
    def quat(self) -> list[float]:
        return [np.cos(self.yaw_rad / 2), 0.0, 0.0, np.sin(self.yaw_rad / 2)]

    def inertia(self, mass: float) -> list[float]:
        hx, hy, hz = self.half
        if self.shape == "cylinder":
            side = mass * (3 * hx**2 + 4 * hz**2) / 12
            return [side, side, mass * hx**2 / 2]
        return [mass * (hy**2 + hz**2) / 3, mass * (hx**2 + hz**2) / 3, mass * (hx**2 + hy**2) / 3]

    def resting_pos(self, obj: DetectedObject, layout: "TwinLayout") -> tuple[float, float, float]:
        x, y = layout.mat_to_world(obj.x_mm, obj.y_mm)
        return x, y, self.half[2] + 1e-3


@dataclass(frozen=True)
class BlockPhysics:
    mass_kg: float = 0.05
    friction: float = 1.0  # sliding coefficient; torsional/rolling stay small


class TwinRenderer:
    """One GL context for many twins that share meshes, i.e. variants copied from one compiled scene.

    `mjv_updateScene` takes each twin's own model, so per-variant cube sizes still render correctly;
    a context per twin made context switches dominate rendering.
    """

    def __init__(self, model: mujoco.MjModel, size: tuple[int, int]) -> None:
        self.renderer = mujoco.Renderer(model, *size)
        self.option, self.perturb, self.camera = mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def render(self, model: mujoco.MjModel, data: mujoco.MjData, camera: str) -> np.ndarray:
        self.camera.fixedcamid = model.camera(camera).id
        mujoco.mjv_updateScene(
            model, data, self.option, self.perturb, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.renderer.scene
        )
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


class Fixture(Protocol):
    name: str

    def add(self, world: mujoco.MjsBody, layout: TwinLayout) -> None: ...


@dataclass
class Twin:
    """MuJoCo SO-101 scene rebuilt from perceived objects on the mat."""

    model: mujoco.MjModel
    data: mujoco.MjData
    blocks: dict[str, DetectedObject]
    fixtures: tuple[Fixture, ...] = ()  # `twin.tasks`: static or free bodies beside the blocks, never re-sized
    renderers: dict[tuple[int, int], TwinRenderer] = field(default_factory=dict, repr=False)

    @classmethod
    def from_objects(
        cls,
        objects: list[DetectedObject],
        layout: TwinLayout | None = None,
        physics: BlockPhysics | None = None,
        start_pose: np.ndarray | None = None,
        so101_xml: Path = SO101_XML,
        look: SceneLook | None = None,
        fixtures: tuple[Fixture, ...] = (),
    ) -> "Twin":
        """Blocks are named by label (suffixed when a colour repeats), so task checks don't depend on order. `fixtures`
        add their own bodies (a basket, a peg, a ring) before the scene is compiled."""
        layout, look = layout or TwinLayout(), look or SceneLook()
        spec = mujoco.MjSpec.from_file(str(so101_xml))
        for material in spec.materials:
            material.rgba = SERVO_RGBA if material.name.startswith("sts3215") else ARM_PRINT_RGBA
        world = spec.worldbody
        spec.visual.headlight.ambient = [look.ambient] * 3
        world.add_light(
            pos=[0.2, 0, 1.5],
            dir=list(look.light_dir),
            diffuse=[look.light_diffuse] * 3,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        floor = world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[*[look.floor_grey] * 3, 1.0])
        if look.floor_texture:
            cls.texture_floor(spec, floor, look.floor_texture)
        if look.wrist_camera:
            cls.move_wrist_camera(spec, look.wrist_camera)
        blocks = dict(zip(cls.block_names(objects), objects))
        for name, obj in blocks.items():
            cls.add_block(world, name, obj, layout, physics or BlockPhysics())
        for fixture in fixtures:
            fixture.add(world, layout)
        world.add_camera(name="front_cam", **(look.camera or layout.front_camera_spec()))  # checkpoint's "front"
        model = spec.compile()
        twin = cls(model, mujoco.MjData(model), blocks, tuple(fixtures))
        mujoco.mj_forward(model, twin.data)  # poses (block_pos, cameras) are valid before the first step
        if start_pose is not None:
            twin.reset_arm(start_pose)
            twin.settle(0.3)
        return twin

    @staticmethod
    def move_wrist_camera(spec: mujoco.MjSpec, pose: dict[str, Any]) -> None:
        """Put the wrist camera at `pose`. Off the menagerie mount its bracket would fill the view, so the bracket is
        hidden (visual only: its collision boxes are untouched)."""
        camera = spec.camera("wrist_cam")
        camera.pos, camera.quat, camera.fovy = pose["pos"], pose["quat"], pose["fovy"]
        if pose.get("mount") != "menagerie":
            for geom in spec.body("camera_mount").geoms:
                geom.rgba = [*geom.rgba[:3], 0.0]

    @staticmethod
    def texture_floor(spec: mujoco.MjSpec, floor: mujoco.MjsGeom, name: str) -> None:
        """Paint the floor with a procedural texture, tiled every 20 cm whatever the camera."""
        tex = spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, width=SIZE, height=SIZE, nchannel=3)
        tex.data = texture(name)
        material = spec.add_material(name="floor", texrepeat=[5, 5], texuniform=True)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
        floor.material, floor.rgba = "floor", [1.0, 1.0, 1.0, 1.0]

    @staticmethod
    def block_names(objects: list[DetectedObject]) -> list[str]:
        """The label, or label_<k> for the k-th object of a colour that appears more than once."""
        counts = Counter(obj.label for obj in objects)
        seen: Counter[str] = Counter()
        names = []
        for obj in objects:
            names.append(obj.label if counts[obj.label] == 1 else f"{obj.label}_{seen[obj.label]}")
            seen[obj.label] += 1
        return names

    def geometry(self, name: str) -> BlockGeometry:
        return BlockGeometry.of(self.blocks[name])

    @staticmethod
    def add_block(
        world: mujoco.MjsBody, name: str, obj: DetectedObject, layout: TwinLayout, physics: BlockPhysics
    ) -> None:
        g = BlockGeometry.of(obj)
        body = world.add_body(name=name, pos=list(g.resting_pos(obj, layout)), quat=g.quat)
        body.add_freejoint(name=name)
        body.add_geom(
            name=name,
            type=g.geom_type,
            size=g.size,
            mass=physics.mass_kg,
            rgba=(*np.multiply(obj.rgb, FLOOR_RGBA[:3]), 1.0) if obj.rgb else RGBA.get(obj.label, (0.5, 0.5, 0.5, 1)),
            friction=[physics.friction, 0.005, 0.0001],
        )

    def variant(
        self, objects: list[DetectedObject], layout: TwinLayout, physics: BlockPhysics, start_pose: np.ndarray
    ) -> "Twin":
        """Same scene with re-sized, re-weighted, moved blocks, without recompiling the MJCF (STL hulls dominate).

        The model is copied and its block parameters edited in place; `objects` must match the template's
        labels and shapes, in the same order.
        """
        model = copy.copy(self.model)
        data = mujoco.MjData(model)
        blocks = dict(zip(self.block_names(objects), objects))
        for name, obj in blocks.items():
            g = BlockGeometry.of(obj)
            geom, body, joint = model.geom(name), model.body(name), model.joint(name)
            if geom.type[0] != g.geom_type:
                raise ValueError(f"{name} is a {g.shape} here but not in the template")
            geom.size[:] = g.size
            geom.rbound[0] = g.rbound
            model.geom_aabb[geom.id, 3:] = (g.half[0], g.half[0] if g.shape == "cylinder" else g.half[1], g.half[2])
            geom.friction[0] = physics.friction
            body.mass[0] = physics.mass_kg
            body.inertia[:] = g.inertia(physics.mass_kg)
            adr = joint.qposadr[0]
            body.pos[:] = model.qpos0[adr : adr + 3] = g.resting_pos(obj, layout)
            body.quat[:] = model.qpos0[adr + 3 : adr + 7] = g.quat
        mujoco.mj_setConst(model, data)  # rebuild constants derived from the edited mass/size
        mujoco.mj_resetData(model, data)  # start from the edited qpos0
        twin = type(self)(model, data, blocks, self.fixtures)
        twin.reset_arm(start_pose)
        twin.settle(0.3)
        return twin

    @property
    def fixture_names(self) -> list[str]:
        return [f.name for f in self.fixtures]

    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.joint(j).qpos[0] for j in ARM_JOINTS])

    def arm_limits(self) -> np.ndarray:
        """(6, 2) joint ranges in radians, which the LeRobot calibration maps onto its normalised units."""
        return np.array([self.model.joint(j).range for j in ARM_JOINTS])

    def to_lerobot(self, qpos: np.ndarray) -> np.ndarray:
        lo, hi = self.arm_limits().T
        unit = (qpos - lo) / (hi - lo)
        return np.concatenate([unit[:5] * 200 - 100, unit[5:] * 100])

    def from_lerobot(self, joints: np.ndarray) -> np.ndarray:
        return lerobot_to_qpos(joints, self.arm_limits())

    def reset_arm(self, qpos: np.ndarray) -> None:
        for joint, q in zip(ARM_JOINTS, qpos):
            self.data.joint(joint).qpos[0] = q
        self.data.ctrl[: len(ARM_JOINTS)] = qpos
        mujoco.mj_forward(self.model, self.data)

    def step_towards(self, qpos_target: np.ndarray, seconds: float) -> None:
        self.data.ctrl[: len(ARM_JOINTS)] = qpos_target
        self.settle(seconds)

    def settle(self, seconds: float = 0.5) -> None:
        mujoco.mj_step(self.model, self.data, nstep=round(seconds / self.model.opt.timestep))

    def block_pos(self, name: str) -> np.ndarray:
        """A block's or a fixture's position (world frame, metres)."""
        if name not in self.blocks and name not in self.fixture_names:  # repeated colours are suffixed (red_0, red_1)
            raise KeyError(f"no block {name!r}; the twin has {list(self.blocks)} and fixtures {self.fixture_names}")
        return self.data.body(name).xpos.copy()

    def project(self, world: np.ndarray, camera: str, size: tuple[int, int] = (480, 640)) -> np.ndarray:
        """Pixel (u, v) of world points (N, 3) in `camera`'s render at `size` (height, width)."""
        mujoco.mj_forward(self.model, self.data)
        cam = self.model.camera(camera).id
        h, w = size
        f = (h / 2) / np.tan(np.radians(self.model.cam_fovy[cam]) / 2)
        local = (world - self.data.cam_xpos[cam]) @ self.data.cam_xmat[cam].reshape(3, 3)  # x right, y up, looks -z
        depth = np.where(local[:, 2] < 0, -local[:, 2], np.nan)  # behind the camera: no pixel
        return np.stack([w / 2 + f * local[:, 0] / depth, h / 2 - f * local[:, 1] / depth], axis=1)

    def hide_arm(self) -> None:
        """Make everything but the blocks and the floor invisible, for comparing objects with an arm-free photo."""
        blocks = {self.model.body(name).id for name in (*self.blocks, *self.fixture_names)}
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] not in blocks and self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                self.model.geom_rgba[g, 3] = 0.0

    def render(
        self, camera: str, size: tuple[int, int] = (480, 640), renderer: TwinRenderer | None = None
    ) -> np.ndarray:
        """Uses a shared `renderer` if given, else this twin's own (one per image size)."""
        if renderer is None:
            if size not in self.renderers:
                self.renderers[size] = TwinRenderer(self.model, size)
            renderer = self.renderers[size]
        return renderer.render(self.model, self.data, camera)

    def observe(
        self, task: str, size: tuple[int, int] = (480, 640), renderer: TwinRenderer | None = None
    ) -> Observation:
        front, wrist = (self.render(f"{cam}_cam", size, renderer) for cam in CAMERA_KEYS)
        return Observation(front=front, wrist=wrist, joints=self.to_lerobot(self.arm_qpos()), task=task)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
