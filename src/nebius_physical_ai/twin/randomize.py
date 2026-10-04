import colorsys
from dataclasses import dataclass, replace

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.scene import FLOOR_RGBA, RGBA, SceneLook, TwinLayout

DISTRACTOR_LABELS = ("yellow", "green", "purple", "dark")
TASK_LABELS = ("blue", "red")  # the expert and StackCheck look these up by name, so nothing else may use them


def radius(obj: DetectedObject) -> float:
    """Half the footprint's diagonal: a circle every orientation of the object fits in."""
    side = float(np.sqrt(obj.area_mm2))
    return float(np.hypot(obj.length_mm or side, obj.width_mm or side)) / 2


def outline(obj: DetectedObject, centre: np.ndarray, n: int = 32) -> np.ndarray:
    """Points around the object's footprint at `centre` (mat mm, its own yaw): a rectangle, or a circle."""
    side = float(np.sqrt(obj.area_mm2))
    half = np.array([obj.length_mm or side, obj.width_mm or side]) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    unit = np.stack([np.cos(t), np.sin(t)], 1)
    local = unit * half if obj.shape == "cylinder" else unit / np.abs(unit / half).max(1, keepdims=True)
    yaw = np.radians(obj.yaw_deg)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return centre + local @ rot.T


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in `p` (..., 2) to the segment a-b."""
    t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1)


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
    pick_length_mm: float = 36.0  # a pick box's long side, so it fits the jaws whichever way they close
    pick_height_mm: tuple[float, float] = (18.0, 40.0)
    place_width_mm: tuple[float, float] = (30.0, 60.0)
    place_height_mm: tuple[float, float] = (15.0, 32.0)  # pick on top stays under the 75 mm carry height
    distractor_height_mm: tuple[float, float] = (18.0, 45.0)
    hue_jitter: float = 0.03
    distractors: tuple[int, int] = (0, 2)
    clearance_mm: float = 15.0  # between footprints
    corridor_mm: float = 45.0  # free band either side of the arm's paths: base -> pick, base -> place, pick -> place


class SceneRandomizer:
    """Per-episode looks (camera, light, mat) and objects (shapes, sizes, colours, distractors) for demos."""

    def __init__(
        self,
        layout: TwinLayout,
        seed: int = 0,
        ranges: Randomization | None = None,
        real_objects: list[DetectedObject] | None = None,
    ) -> None:
        """`real_objects` (e.g. `npai lift` output) are placed as distractors half the time, so demos see real clutter."""
        self.layout, self.ranges = layout, ranges or Randomization()
        self.real_objects = [o for o in real_objects or [] if o.label not in TASK_LABELS]
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

    def block(
        self,
        label: str,
        width_mm: tuple[float, float],
        height_mm: tuple[float, float],
        max_length_mm: float = float("inf"),
        xy: tuple[float, float] = (0.0, 0.0),
    ) -> DetectedObject:
        width = self.uniform(width_mm)
        cylinder = self.rng.random() < 0.5
        length = width if cylinder else min(width * self.rng.uniform(1.0, 1.3), max(max_length_mm, width))
        return DetectedObject(
            label,
            round(xy[0], 1),
            round(xy[1], 1),
            round(length * width, 1),
            length_mm=round(length, 1),
            width_mm=round(width, 1),
            yaw_deg=round(self.rng.uniform(-90, 90), 1),
            height_mm=round(self.uniform(height_mm), 1),
            shape="cylinder" if cylinder else "box",
            rgb=self.colour(label),
        )

    def objects(self, sampler: SceneSampler) -> list[DetectedObject]:
        """The sampler's pick (blue) and place (red) positions as random shapes, plus distractors clear of the arm."""
        pick, place = sampler.sample()
        r = self.ranges
        found = [
            self.block("blue", r.pick_width_mm, r.pick_height_mm, r.pick_length_mm, (pick.x_mm, pick.y_mm)),
            self.block("red", r.place_width_mm, r.place_height_mm, xy=(place.x_mm, place.y_mm)),
        ]
        for _ in range(int(self.rng.integers(r.distractors[0], r.distractors[1] + 1))):
            candidate = self.distractor()
            if (xy := self.free_spot(found, candidate)) is not None:
                found.append(replace(candidate, x_mm=round(float(xy[0]), 1), y_mm=round(float(xy[1]), 1)))
        return found

    def distractor(self) -> DetectedObject:
        """A copy of a real object (size +-10%) half the time when there are any, else a random shape."""
        if self.real_objects and self.rng.random() < 0.5:
            real = self.real_objects[int(self.rng.integers(len(self.real_objects)))]
            s = self.rng.uniform(0.9, 1.1)
            return replace(
                real,
                area_mm2=round(real.area_mm2 * s * s, 1),
                length_mm=round(real.length_mm * s, 1),
                width_mm=round(real.width_mm * s, 1),
                height_mm=round(real.height_mm * s, 1),
                yaw_deg=round(self.rng.uniform(-90, 90), 1),
            )
        label = str(self.rng.choice(DISTRACTOR_LABELS))
        return self.block(label, self.ranges.place_width_mm, self.ranges.distractor_height_mm)

    def free_spot(self, taken: list[DetectedObject], candidate: DetectedObject, tries: int = 100) -> np.ndarray | None:
        """A centre on the mat where the candidate's own outline clears every object and the arm's corridors."""
        mat, ranges = self.layout.mat, self.ranges
        base = np.array([mat.width_mm / 2, mat.height_mm + self.layout.base_offset_mm])
        pick, place = (np.array([o.x_mm, o.y_mm]) for o in taken[:2])
        corridors = [(base, pick), (base, place), (pick, place)]
        for _ in range(tries):
            centre = self.rng.uniform([0, 0], [mat.width_mm, mat.height_mm])
            edge = outline(candidate, centre)
            clear = all(np.hypot(*(edge - [o.x_mm, o.y_mm]).T).min() > radius(o) + ranges.clearance_mm for o in taken)
            if clear and all(segment_distance(edge, a, b).min() > ranges.corridor_mm for a, b in corridors):
                return centre
        return None
