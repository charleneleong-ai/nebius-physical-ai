import colorsys
from dataclasses import dataclass

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.scene import FLOOR_RGBA, RGBA, SceneLook, TwinLayout

DISTRACTOR_LABELS = ("yellow", "green", "purple", "dark")


@dataclass(frozen=True)
class Randomization:
    """Ranges for demo randomisation: wide enough that the policy can't lean on any one camera, light or object."""

    elevation_deg: tuple[float, float] = (35.0, 80.0)  # camera above the mat plane
    azimuth_deg: tuple[float, float] = (-130.0, 130.0)  # 0 = across the mat facing the robot; behind the arm is out
    distance_mm: tuple[float, float] = (350.0, 650.0)  # camera to its aim point
    fovy_deg: tuple[float, float] = (32.0, 52.0)  # vertical, at the 4:3 render
    aim_jitter_mm: float = 40.0  # aim point around the mat centre
    light_tilt_deg: tuple[float, float] = (0.0, 35.0)  # steeper grazing light makes shadow-map acne
    light_diffuse: tuple[float, float] = (0.35, 0.8)
    ambient: tuple[float, float] = (0.05, 0.25)
    floor_grey: tuple[float, float] = (0.45, 0.7)  # brighter saturates to white under the key light
    pick_width_mm: tuple[float, float] = (22.0, 34.0)  # the gripper's comfortable range
    place_width_mm: tuple[float, float] = (30.0, 60.0)
    height_mm: tuple[float, float] = (18.0, 45.0)
    hue_jitter: float = 0.03
    distractors: tuple[int, int] = (0, 2)
    distractor_gap_mm: float = 70.0


class SceneRandomizer:
    """Per-episode looks (camera, light, mat) and objects (shapes, sizes, colours, distractors) for demos."""

    def __init__(self, layout: TwinLayout, seed: int = 0, ranges: Randomization | None = None) -> None:
        self.layout, self.ranges = layout, ranges or Randomization()
        self.rng = np.random.default_rng(seed)

    def uniform(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(*bounds))

    def look(self) -> SceneLook:
        r = self.ranges
        tilt, heading = np.radians(self.uniform(r.light_tilt_deg)), self.rng.uniform(0, 2 * np.pi)
        light = (np.sin(tilt) * np.cos(heading), np.sin(tilt) * np.sin(heading), -np.cos(tilt))
        return SceneLook(
            camera=self.camera(),
            light_dir=tuple(float(v) for v in light),
            light_diffuse=self.uniform(r.light_diffuse),
            ambient=self.uniform(r.ambient),
            floor_grey=self.uniform(r.floor_grey),
        )

    def camera(self) -> dict[str, list[float] | float]:
        """A camera on an arc around the mat (never behind the arm), aimed near its centre, image up = world up-ish."""
        r, mat = self.ranges, self.layout.mat
        aim_xy = self.layout.mat_to_world(mat.width_mm / 2, mat.height_mm / 2)
        aim = np.array([*aim_xy, 0.0]) + np.append(self.rng.normal(0, r.aim_jitter_mm / 1000, 2), 0.0)
        elevation, azimuth = np.radians(self.uniform(r.elevation_deg)), np.radians(self.uniform(r.azimuth_deg))
        offset = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        position = aim + self.uniform(r.distance_mm) / 1000 * offset
        forward = -offset
        up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward
        up /= np.linalg.norm(up)
        right = np.cross(forward, up)  # MuJoCo cameras look down -z with y up: x = forward × up
        return {"pos": position.tolist(), "xyaxes": [*right.tolist(), *up.tolist()], "fovy": self.uniform(r.fovy_deg)}

    def colour(self, label: str) -> tuple[float, float, float]:
        """The label's palette colour with hue, saturation and brightness jittered, relative to the mat's paper."""
        h, s, v = colorsys.rgb_to_hsv(*RGBA.get(label, (0.5, 0.5, 0.5, 1))[:3])
        h = (h + self.rng.uniform(-self.ranges.hue_jitter, self.ranges.hue_jitter)) % 1.0
        rgb = colorsys.hsv_to_rgb(h, s * self.rng.uniform(0.7, 1.0), v * self.rng.uniform(0.6, 1.0))
        return tuple(round(float(np.clip(c / FLOOR_RGBA[0], 0, 1)), 3) for c in rgb)

    def block(self, label: str, x_mm: float, y_mm: float, width_mm: tuple[float, float]) -> DetectedObject:
        width = self.uniform(width_mm)
        cylinder = self.rng.random() < 0.5
        length = width if cylinder else width * self.rng.uniform(1.0, 1.3)
        return DetectedObject(
            label,
            round(x_mm, 1),
            round(y_mm, 1),
            round(length * width, 1),
            length_mm=round(length, 1),
            width_mm=round(width, 1),
            yaw_deg=round(self.rng.uniform(-90, 90), 1),
            height_mm=round(self.uniform(self.ranges.height_mm), 1),
            shape="cylinder" if cylinder else "box",
            rgb=self.colour(label),
        )

    def objects(self, sampler: SceneSampler) -> list[DetectedObject]:
        """The sampler's pick (blue) and place (red) positions as random shapes, plus distractors elsewhere."""
        pick, place = sampler.sample()
        found = [
            self.block("blue", pick.x_mm, pick.y_mm, self.ranges.pick_width_mm),
            self.block("red", place.x_mm, place.y_mm, self.ranges.place_width_mm),
        ]
        lo, hi = self.ranges.distractors
        for _ in range(int(self.rng.integers(lo, hi + 1))):
            if (xy := self.free_spot(found)) is not None:
                found.append(self.block(str(self.rng.choice(DISTRACTOR_LABELS)), *xy, self.ranges.place_width_mm))
        return found

    def free_spot(self, taken: list[DetectedObject], tries: int = 50) -> tuple[float, float] | None:
        mat, margin = self.layout.mat, 40.0
        for _ in range(tries):
            xy = self.rng.uniform([margin, margin], [mat.width_mm - margin, mat.height_mm - margin])
            if all(np.hypot(xy[0] - o.x_mm, xy[1] - o.y_mm) > self.ranges.distractor_gap_mm for o in taken):
                return float(xy[0]), float(xy[1])
        return None
