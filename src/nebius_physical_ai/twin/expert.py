from collections.abc import Iterator
from dataclasses import dataclass

import mujoco
import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin.rollout import CONTROL_HZ
from nebius_physical_ai.twin.scene import ARM_JOINTS, Twin, TwinLayout

GRIP_OPEN, GRIP_CLOSED = 1.2, 0.1  # radians; closed < 0.2 holds a 30 mm cube
# Fingertip heights; top-down reach tops out near 7.5 cm, so hover low rather than above the cubes' height.
PICK_HOVER_Z, PLACE_HOVER_Z = 0.06, 0.075
REST_POSE = np.array([0.0, -0.5, 0.5, 1.0, 0.0])  # arm folded above the base
# Staged mistakes for recovery demos, matching v5's commonest failures (no contact or a push, and a drop).
MISTAKES = ("miss", "drop")
MISS_OFFSET_MM = (8.0, 25.0)  # under ~15 mm the jaws land on the object and push it
DROP_FRACTION = (0.2, 0.8)  # of the way from pick to place where the object is let go
Segment = tuple[np.ndarray, float, float]  # arm joint target, gripper target, seconds


class UnreachableTarget(ValueError):
    """The top-down IK can't reach a waypoint within tolerance."""


def start_pose(rng: np.random.Generator, sigma_rad: float = 0.05) -> np.ndarray:
    """Rest pose with joint noise and the gripper open: where demos and rehearsals begin."""
    return np.append(REST_POSE + rng.normal(0, sigma_rad, 5), GRIP_OPEN)


class TopDownIK:
    """Damped least squares on the gripper site: position plus fingers (site x axis) pointing down.

    Tries progressively weaker orientation weights, so edge-of-workspace targets trade a little tilt for position.
    """

    ROT_WEIGHTS = (0.3, 0.1, 0.03)

    def __init__(self, model: mujoco.MjModel, iters: int = 300, tol_m: float = 0.002) -> None:
        self.model, self.iters, self.tol_m = model, iters, tol_m
        self.site = model.site("gripperframe").id
        arm = ARM_JOINTS[:5]
        self.dofs = [model.joint(j).dofadr[0] for j in arm]
        self.qadr = [model.joint(j).qposadr[0] for j in arm]
        self.lo, self.hi = np.array([model.joint(j).range for j in arm]).T
        self._data = mujoco.MjData(model)

    def solve(self, target: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (5 arm joint angles, position error in metres) from the strictest weight within tolerance."""
        attempts = []
        for w in self.ROT_WEIGHTS:
            q, err = self.solve_weighted(target, q0, w)
            if err <= self.tol_m:
                return q, err
            attempts.append((q, err))
        return min(attempts, key=lambda a: a[1])

    def solve_weighted(self, target: np.ndarray, q0: np.ndarray, w: float) -> tuple[np.ndarray, float]:
        m, d = self.model, self._data
        q, jp, jr = q0.copy(), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(self.iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_pos = target - d.site_xpos[self.site]
            e_rot = np.cross(d.site_xmat[self.site].reshape(3, 3)[:, 0], [0.0, 0.0, -1.0])
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            jac = np.vstack([jp[:, self.dofs], w * jr[:, self.dofs]])
            err = np.concatenate([e_pos, w * e_rot])
            q = np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
        return q, float(np.linalg.norm(e_pos))


@dataclass
class Frame:
    obs: Observation
    action: np.ndarray  # commanded joints, LeRobot units


class PickPlaceExpert:
    """Scripted top-down pick of `pick` and place onto `place`, interpolated in joint space at CONTROL_HZ."""

    def __init__(
        self, twin: Twin, task: str, pick: str = "blue", place: str = "red", image_size: tuple[int, int] = (480, 640)
    ) -> None:
        self.twin, self.task, self.pick, self.place, self.image_size = twin, task, pick, place, image_size
        self.ik = TopDownIK(twin.model)

    def waypoints(self) -> list[Segment]:
        pick = self.twin.block_pos(self.pick)
        place = self.twin.block_pos(self.place)
        stack_z = place[2] + self.twin.geometry(self.place).half[2] + self.twin.geometry(self.pick).half[2]
        return self.pick_place(pick, np.array([place[0], place[1], stack_z + 0.005]))

    def pick_place(self, pick: np.ndarray, release: np.ndarray, hover_z: float = PLACE_HOVER_Z) -> list[Segment]:
        """Lift from `pick`, carry over `hover_z` and open at `release`: the grasp sequence every task shares."""
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

    def solve(self, target: np.ndarray, name: str, seed: np.ndarray = REST_POSE) -> np.ndarray:
        q, err = self.ik.solve(target, seed)
        if err > 0.005:
            raise UnreachableTarget(f"{name} unreachable ({err * 1000:.1f} mm)")
        return q

    @staticmethod
    def grasp(hover: np.ndarray, low: np.ndarray) -> list[Segment]:
        """Hover, descend open, close, lift."""
        return [(hover, GRIP_OPEN, 1.5), (low, GRIP_OPEN, 1.0), (low, GRIP_CLOSED, 0.8), (hover, GRIP_CLOSED, 1.0)]

    def mistake(self, kind: str, rng: np.random.Generator) -> list[Segment]:
        """Segments that fail the way rehearsals do: `miss` closes the jaws 8-25 mm off the pick object (often
        nudging it), `drop` lifts it and lets go part of the way to the place object."""
        pick, place = self.twin.block_pos(self.pick), self.twin.block_pos(self.place)
        if kind == "miss":
            angle, off = rng.uniform(0, 2 * np.pi), rng.uniform(*MISS_OFFSET_MM) / 1000
            at = pick + off * np.array([np.cos(angle), np.sin(angle), 0.0])
            return self.grasp(self.solve(np.array([at[0], at[1], PICK_HOVER_Z]), "miss_hover"), self.solve(at, "miss"))
        if kind == "drop":
            hover, low = self.solve(np.array([pick[0], pick[1], PICK_HOVER_Z]), "pick_hover"), self.solve(pick, "pick")
            way = pick[:2] + rng.uniform(*DROP_FRACTION) * (place[:2] - pick[:2])
            carry = self.solve(np.array([way[0], way[1], PICK_HOVER_Z]), "drop")
            return [*self.grasp(hover, low), (carry, GRIP_CLOSED, 1.0), (carry, GRIP_OPEN, 0.5)]
        raise ValueError(f"unknown mistake {kind!r}; expected one of {MISTAKES}")

    def stage(self, kind: str, rng: np.random.Generator) -> None:
        """Play a `mistake` without recording it, so the demo that follows starts from a failed rehearsal's state
        and shows only the recovery (the policy never learns the mistake itself)."""
        for _ in self.follow(self.mistake(kind, rng)):
            pass
        self.twin.settle(0.5)

    @staticmethod
    def targets(segments: list[Segment], start: np.ndarray) -> Iterator[np.ndarray]:
        """The joint target for each control tick through `segments`, interpolated from `start`."""
        for arm, grip, seconds in segments:
            goal = np.append(arm, grip)
            yield from np.linspace(start, goal, max(1, round(seconds * CONTROL_HZ)) + 1)[1:]
            start = goal

    def follow(self, segments: list[Segment]) -> Iterator[np.ndarray]:
        """Drive through `segments` from wherever the arm is, yielding each commanded target before stepping to it."""
        for target in self.targets(segments, self.twin.arm_qpos()):
            yield target
            self.twin.step_towards(target, 1 / CONTROL_HZ)

    def run(self) -> Iterator[Frame]:
        for target in self.follow(self.waypoints()):
            yield Frame(self.twin.observe(self.task, self.image_size), self.twin.to_lerobot(target))


class SceneSampler:
    """Random pick/place cube layouts inside the arm's top-down workspace (16–22 cm out, ±9 cm sideways)."""

    def __init__(
        self,
        layout: TwinLayout,
        seed: int | list[int] = 0,
        min_gap_mm: float = 60.0,
        reach_mm: tuple[float, float] = (160.0, 220.0),
        half_width_mm: float = 90.0,
    ) -> None:
        """`reach_mm` is the distance in front of the base; `half_width_mm` the sideways spread."""
        self.layout, self.min_gap_mm = layout, min_gap_mm
        self.rng = np.random.default_rng(seed)
        mat = layout.mat
        base_y = mat.height_mm + layout.base_offset_mm
        self.y_range = (base_y - reach_mm[1], base_y - reach_mm[0])
        self.x_range = (mat.width_mm / 2 - half_width_mm, mat.width_mm / 2 + half_width_mm)

    @classmethod
    def hard(cls, layout: TwinLayout, seed: int = 0) -> "SceneSampler":
        """Cubes as close as 35 mm, sampled out to the edges of reach: where the fine-tuned policy struggles."""
        return cls(layout, seed, min_gap_mm=35.0, reach_mm=(150.0, 245.0), half_width_mm=115.0)

    def sample(self, labels: tuple[str, str] = ("blue", "red"), side_mm: float = 30.0) -> list[DetectedObject]:
        while True:
            xy = self.rng.uniform([self.x_range[0], self.y_range[0]], [self.x_range[1], self.y_range[1]], (2, 2))
            if np.linalg.norm(xy[0] - xy[1]) >= self.min_gap_mm:
                return [DetectedObject(label, *np.round(p, 1), side_mm**2) for label, p in zip(labels, xy)]

    def start_pose(self) -> np.ndarray:
        return start_pose(self.rng)
