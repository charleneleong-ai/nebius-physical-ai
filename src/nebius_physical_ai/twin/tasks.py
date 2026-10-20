"""Tasks other than stacking, for the held_out_tasks suite: tasks no checkpoint here was trained on, after ArmnetBench.

ArmnetBench's eight SO-101 tasks (armnet/armnetbench_v01_lerobot_so101) are mostly beyond a rigid-body twin (cables,
tool slots), but two map onto it: putting eye drops into a basket and putting a ring over a peg. Each task adds static
or free `Fixture`s to the scene beside the ordinary blocks, a success check, and the points a scripted top-down pick
and place needs (`TaskExpert`), so the benchmark keeps only scenes the expert can actually solve.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import mujoco
import numpy as np

from nebius_physical_ai.twin.expert import (
    GRIP_CLOSED,
    GRIP_OPEN,
    PICK_HOVER_Z,
    PLACE_HOVER_Z,
    REST_POSE,
    PickPlaceExpert,
    Segment,
)
from nebius_physical_ai.twin.rollout import Check
from nebius_physical_ai.twin.scene import Twin, TwinLayout

WICKER = (0.62, 0.45, 0.25, 1.0)
WOOD = (0.55, 0.38, 0.22, 1.0)


@dataclass(frozen=True)
class Basket:
    """An open-topped box fixed to the table: a floor and four walls, one static body named `name`."""

    x_mm: float
    y_mm: float
    inner_mm: float = 80.0
    wall_mm: float = 35.0
    thick_mm: float = 4.0
    name: str = "basket"

    def add(self, world: mujoco.MjsBody, layout: TwinLayout) -> None:
        body = world.add_body(name=self.name, pos=[*layout.mat_to_world(self.x_mm, self.y_mm), 0.0])
        half, t, h = self.inner_mm / 2000, self.thick_mm / 2000, self.wall_mm / 2000
        body.add_geom(name=f"{self.name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half + 2 * t, half + 2 * t, t],
                      pos=[0, 0, t], rgba=WICKER)  # fmt: skip
        for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            size = [t, half + 2 * t, h] if dx else [half + 2 * t, t, h]
            body.add_geom(name=f"{self.name}_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                          pos=[dx * (half + t), dy * (half + t), h], rgba=WICKER)  # fmt: skip

    def contains(self, twin: Twin, obj: str) -> bool:
        """The object's centre is inside the walls and below the rim."""
        x, y, z = twin.block_pos(obj) - twin.data.body(self.name).xpos
        return bool(max(abs(x), abs(y)) < self.inner_mm / 2000 and z < self.wall_mm / 1000)


@dataclass(frozen=True)
class Peg:
    """A vertical wooden peg fixed to the table."""

    x_mm: float
    y_mm: float
    radius_mm: float = 8.0
    height_mm: float = 45.0
    name: str = "peg"

    def add(self, world: mujoco.MjsBody, layout: TwinLayout) -> None:
        body = world.add_body(name=self.name, pos=[*layout.mat_to_world(self.x_mm, self.y_mm), 0.0])
        body.add_geom(name=self.name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[self.radius_mm / 1000, self.height_mm / 2000], pos=[0, 0, self.height_mm / 2000],
                      rgba=WOOD)  # fmt: skip


@dataclass(frozen=True)
class Ring:
    """A flat ring free on the table, its tube approximated by `segments` capsules around the centreline."""

    x_mm: float
    y_mm: float
    radius_mm: float = 22.0  # centreline
    tube_mm: float = 5.0  # tube radius
    mass_kg: float = 0.03
    segments: int = 12
    rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.55, 1.0)
    name: str = "ring"

    def add(self, world: mujoco.MjsBody, layout: TwinLayout) -> None:
        r, tube = self.radius_mm / 1000, self.tube_mm / 1000
        body = world.add_body(name=self.name, pos=[*layout.mat_to_world(self.x_mm, self.y_mm), tube + 1e-3])
        body.add_freejoint(name=self.name)
        angles = np.linspace(0, 2 * np.pi, self.segments + 1)
        for k, (a, b) in enumerate(pairwise(angles)):
            body.add_geom(name=f"{self.name}_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[tube, 0, 0],
                          fromto=[r * np.cos(a), r * np.sin(a), 0, r * np.cos(b), r * np.sin(b), 0],
                          mass=self.mass_kg / self.segments, rgba=self.rgba, friction=[1.0, 0.005, 0.0001])  # fmt: skip

    @property
    def hole_m(self) -> float:
        return (self.radius_mm - self.tube_mm) / 1000


class BasketCheck:
    """Success for 'put <obj> in the basket'. `top`/`bottom` name the moving object and the target for FailureWatch."""

    def __init__(self, obj: str, basket: Basket) -> None:
        self.top, self.bottom, self.basket = obj, basket.name, basket

    def __call__(self, twin: Twin) -> bool:
        return self.basket.contains(twin, self.top)

    def expert_points(self, twin: Twin) -> tuple[np.ndarray, np.ndarray]:
        """Where to grasp, and where to let go: over the basket's centre, the object clear of the rim."""
        target = twin.data.body(self.bottom).xpos
        clear = self.basket.wall_mm / 1000 + twin.geometry(self.top).half[2] + 0.008
        return twin.block_pos(self.top), np.array([target[0], target[1], clear])


class PegCheck:
    """Success for 'put the ring over the peg': the peg passes through the ring's hole and the ring sits below its top."""

    def __init__(self, ring: Ring, peg: Peg, grasp_angle: float = 0.0) -> None:
        self.top, self.bottom, self.ring, self.peg, self.grasp_angle = ring.name, peg.name, ring, peg, grasp_angle

    def __call__(self, twin: Twin) -> bool:
        x, y, z = twin.block_pos(self.top) - twin.data.body(self.bottom).xpos
        return bool(np.hypot(x, y) < self.ring.hole_m - self.peg.radius_mm / 1000 and z < self.peg.height_mm / 1000)

    def expert_points(self, twin: Twin) -> tuple[np.ndarray, np.ndarray]:
        """Grasp the tube at `grasp_angle` (world frame) and let go with that point the same offset from the peg's axis,
        the ring clear of the peg's top."""
        offset = self.ring.radius_mm / 1000 * np.array([np.cos(self.grasp_angle), np.sin(self.grasp_angle), 0.0])
        peg = twin.data.body(self.bottom).xpos
        clear = self.peg.height_mm / 1000 + self.ring.tube_mm / 1000 + 0.008
        return twin.block_pos(self.top) + offset, np.array([peg[0] + offset[0], peg[1] + offset[1], clear])


class TaskCheck(Check, Protocol):
    """A task's check that also tells the scripted expert where to grasp and where to let go."""

    def expert_points(self, twin: Twin) -> tuple[np.ndarray, np.ndarray]: ...


class TaskExpert(PickPlaceExpert):
    """The scripted top-down pick and place, with grasp and release points from the task's check."""

    def __init__(self, twin: Twin, task: str, check: TaskCheck, image_size: tuple[int, int] = (480, 640)) -> None:
        super().__init__(twin, task, check.top, check.bottom, image_size)
        self.check = check

    def waypoints(self) -> list[Segment]:
        pick, release = self.check.expert_points(self.twin)
        hover_z = max(PLACE_HOVER_Z, release[2] + 0.01)
        targets = {
            "pick_hover": np.array([pick[0], pick[1], PICK_HOVER_Z]),
            "pick": pick,
            "place_hover": np.array([release[0], release[1], hover_z]),
            "place": release,
        }
        q, joints = REST_POSE, {}
        for name, target in targets.items():
            joints[name] = q = self.solve(target, name, q)
        return [
            *self.grasp(joints["pick_hover"], joints["pick"]),
            (joints["place_hover"], GRIP_CLOSED, 1.5),
            (joints["place"], GRIP_CLOSED, 1.0),
            (joints["place"], GRIP_OPEN, 0.8),
            (joints["place_hover"], GRIP_OPEN, 1.0),
        ]
