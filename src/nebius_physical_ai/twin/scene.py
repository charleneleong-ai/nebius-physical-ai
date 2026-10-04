import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject, MatSpec
from nebius_physical_ai.policy.gr00t import CAMERA_KEYS, Observation

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
}
BLOCK_SIDE_MM = (20.0, 60.0)
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
A3_MAT = MatSpec(width_mm=350.0, height_mm=227.0)  # the printed A3 sheet; MatSpec defaults to the A4 cut-outs


@dataclass(frozen=True)
class TwinLayout:
    """Robot base sits beyond the mat's bottom edge, centred, facing up the mat.

    World frame = robot base frame: +x forward (towards mat top), +y left (towards mat x=0), z up.
    """

    mat: MatSpec = A3_MAT
    base_offset_mm: float = 50.0

    def mat_to_world(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        base_x, base_y = self.mat.width_mm / 2, self.mat.height_mm + self.base_offset_mm
        return (base_y - y_mm) / 1000, (base_x - x_mm) / 1000


@dataclass(frozen=True)
class BlockPhysics:
    mass_kg: float = 0.05
    friction: float = 1.0  # sliding coefficient; torsional/rolling stay small


@dataclass
class Twin:
    """MuJoCo SO-101 scene rebuilt from perceived objects on the mat."""

    model: mujoco.MjModel
    data: mujoco.MjData
    blocks: dict[str, DetectedObject]
    renderers: dict[tuple[int, int], mujoco.Renderer] = field(default_factory=dict, repr=False)

    @classmethod
    def from_objects(
        cls,
        objects: list[DetectedObject],
        layout: TwinLayout | None = None,
        physics: BlockPhysics | None = None,
        start_pose: np.ndarray | None = None,
        so101_xml: Path = SO101_XML,
    ) -> "Twin":
        """Blocks are named by label, so task checks don't depend on perception order."""
        labels = [obj.label for obj in objects]
        if len(set(labels)) != len(labels):
            raise ValueError(f"block labels must be unique, got {labels}")
        layout = layout or TwinLayout()
        spec = mujoco.MjSpec.from_file(str(so101_xml))
        world = spec.worldbody
        world.add_light(pos=[0.2, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.6, 0.6, 0.6, 1])
        blocks = {}
        for obj in objects:
            cls.add_block(world, obj.label, obj, layout, physics or BlockPhysics())
            blocks[obj.label] = obj
        # The checkpoint's "front" view is a top camera: look straight down on the mat, arm at the bottom.
        top_x, _ = layout.mat_to_world(layout.mat.width_mm / 2, layout.mat.height_mm / 2)
        world.add_camera(name="front_cam", pos=[top_x, 0, 0.6], xyaxes=[0, -1, 0, 1, 0, 0])
        model = spec.compile()
        twin = cls(model, mujoco.MjData(model), blocks)
        if start_pose is not None:
            twin.reset_arm(start_pose)
            twin.settle(0.3)
        return twin

    @staticmethod
    def block_side_m(obj: DetectedObject) -> float:
        return float(np.clip(np.sqrt(obj.area_mm2), *BLOCK_SIDE_MM)) / 1000

    @classmethod
    def add_block(
        cls, world: mujoco.MjsBody, name: str, obj: DetectedObject, layout: TwinLayout, physics: BlockPhysics
    ) -> None:
        half = cls.block_side_m(obj) / 2
        x, y = layout.mat_to_world(obj.x_mm, obj.y_mm)
        body = world.add_body(name=name, pos=[x, y, half + 1e-3])
        body.add_freejoint(name=name)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half] * 3,
            mass=physics.mass_kg,
            rgba=RGBA.get(obj.label, (0.5, 0.5, 0.5, 1)),
            friction=[physics.friction, 0.005, 0.0001],
        )

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
        lo, hi = self.arm_limits().T
        unit = np.concatenate([(joints[:5] + 100) / 200, joints[5:] / 100])
        return lo + np.clip(unit, 0, 1) * (hi - lo)

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
        return self.data.body(name).xpos.copy()

    def render(self, camera: str, size: tuple[int, int] = (480, 640)) -> np.ndarray:
        """GL context creation dominates a render, so keep one renderer per image size."""
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.model, *size)
        renderer = self.renderers[size]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observe(self, task: str, size: tuple[int, int] = (480, 640)) -> Observation:
        front, wrist = (self.render(f"{cam}_cam", size) for cam in CAMERA_KEYS)
        return Observation(front=front, wrist=wrist, joints=self.to_lerobot(self.arm_qpos()), task=task)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
