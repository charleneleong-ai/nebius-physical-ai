import colorsys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nebius_physical_ai.capture.camera import CAMERA_SOURCES, DEFAULT_CAMERA_MIX, CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin import roles
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.rollout import STACK_TASK
from nebius_physical_ai.twin.scene import FLOOR_RGBA, SceneLook, TwinLayout
from nebius_physical_ai.twin.textures import HELD_OUT_SEED, KINDS

DISTRACTOR_LABELS = ("yellow", "green", "purple", "dark")
TASK_LABELS = ("blue", "red")  # the expert and StackCheck look these up by name, so nothing else may use them
REAL_DISTRACTOR_SHARE = 0.5  # of distractors, when real objects are available
MAX_DISTRACTOR_LENGTH_MM = 140.0  # the longest real object (the glasses case)


@dataclass(frozen=True)
class Episode:
    """One demo scene and the task it teaches: the objects, which two to use, and the prompt that names them."""

    objects: list[DetectedObject]
    pick: str  # twin block names, as the expert and StackCheck look them up
    place: str
    prompt: str


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


def rotated(v: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """`v` turned about the unit `axis` by `degrees` (Rodrigues)."""
    a = np.radians(degrees)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))


# Wrist cameras in the gripper's camera-mount frame: the menagerie model's own mount, and the Seeed SO-ARM101 mount
# so101-benchmark uses (measured from its scene against the same gripper body).
MENAGERIE_WRIST = {
    "mount": "menagerie",
    "pos": [-0.00296, 0.05436, -0.0531],
    "quat": [0.95828, -0.28579, -0.00156, 0.00455],
    "fovy": 61.0,
}
SEEED_WRIST = {"mount": "seeed", "pos": [0.0025, 0.07145, -0.00322], "quat": [0.9763, -0.21644, 0.0, 0.0], "fovy": 65.0}


def jittered_mount(
    mount: dict[str, Any], rng: np.random.Generator, pos_m: float, tilt_deg: float, fovy_deg: float
) -> dict[str, Any]:
    """`mount` moved up to `pos_m` per axis, tilted up to `tilt_deg` about the camera's own x axis, fovy jittered."""
    half = np.radians(rng.uniform(-tilt_deg, tilt_deg)) / 2
    w, x, y, z = mount["quat"]
    c, s = np.cos(half), np.sin(half)  # mount * (cos, sin, 0, 0): a turn about the camera's x axis
    quat = [w * c - x * s, w * s + x * c, y * c + z * s, z * c - y * s]
    return {
        "mount": mount["mount"],
        "pos": [round(float(v + rng.uniform(-pos_m, pos_m)), 5) for v in mount["pos"]],
        "quat": [round(float(v), 5) for v in quat],
        "fovy": round(float(mount["fovy"]) + float(rng.uniform(-fovy_deg, fovy_deg)), 2),
    }


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in `p` (..., 2) to the segment a-b."""
    t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1)


@dataclass(frozen=True)
class Randomization:
    """Ranges for demo randomisation: wide enough that the policy can't lean on any one camera, light or object."""

    elevation_deg: tuple[float, float] = (35.0, 80.0)  # camera above the mat plane
    azimuth_deg: tuple[float, float] = (-180.0, 180.0)  # 0 = across the mat facing the robot, 180 = behind the arm
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
    # Jitter around a calibrated camera pose (the design spec's +-3 cm, +-5 degrees): covers calibration error too.
    pose_position_mm: float = 30.0  # per mat axis
    pose_rotation_deg: float = 5.0  # each of yaw, pitch and roll, about the camera's own axes
    pose_hfov_deg: float = 5.0
    # Role-agnostic demos: a share of episodes keep the original blue and red cubes and prompt verbatim, a share of the
    # rest place onto a real object, and clutter covers the real objects' sizes (up to 140 mm long and 65 mm tall).
    original_share: float = 0.25
    real_place_share: float = 1 / 3
    wide_distractors: tuple[int, int] = (0, 4)
    wide_distractor_width_mm: tuple[float, float] = (22.0, 65.0)
    wide_distractor_aspect: tuple[float, float] = (1.0, 2.2)
    wide_distractor_height_mm: tuple[float, float] = (15.0, 65.0)
    # Pick-sized decoys of the pick object's own shape in other colours: only the prompt's colour word tells them
    # apart, so a policy can't get by on "grab the small one" (v5's commonest held-out failure was never touching it).
    lookalikes: tuple[int, int] = (0, 0)
    # Visual domain randomisation: a share of floors get a procedural texture (seeds below the held-out ones), and the
    # wrist camera is drawn between the menagerie and Seeed mounts with jitter. Both off by default.
    texture_share: float = 0.0
    texture_seeds: tuple[int, int] = (0, HELD_OUT_SEED)
    wrist_randomize: bool = False
    seeed_share: float = 0.5
    wrist_jitter_m: float = 0.006
    wrist_tilt_jitter_deg: float = 5.0
    wrist_fovy_jitter_deg: float = 3.0


class SceneRandomizer:
    """Per-episode looks (camera, light, mat) and objects (shapes, sizes, colours, distractors) for demos."""

    def __init__(
        self,
        layout: TwinLayout,
        seed: int | list[int] = 0,
        ranges: Randomization | None = None,
        real_objects: list[DetectedObject] | None = None,
        camera_pose: CameraPose | None = None,
        camera_mix: tuple[float, float, float] = DEFAULT_CAMERA_MIX,
        role_agnostic: bool = False,
        colours: tuple[str, ...] = roles.TRAIN_COLOURS,
        pair: tuple[str, str] | None = None,
        exclude_real: tuple[str, ...] | None = None,
    ) -> None:
        """`real_objects` (e.g. `npai lift` output) are placed as distractors half the time, so demos see real clutter.

        With a calibrated `camera_pose` each episode's camera is drawn from `camera_mix` (shares of pose-centred, wide
        arc and straight-down views, as in `CAMERA_SOURCES`); without one it is always the wide arc.

        With `role_agnostic`, `episode` names the pick and place objects by what they look like (`green can`, with the
        prompt to match) in all but `ranges.original_share` of episodes. Their colours come from `colours`; real
        objects named in `exclude_real` (default: the held-out ones) are never used, so a training randomiser can't see
        what the benchmark holds out.
        """
        self.layout, self.ranges = layout, ranges or Randomization()
        self.role_agnostic, self.colours = role_agnostic, colours
        if pair and not set(pair) <= set(colours):
            raise ValueError(f"pinned pair {pair} is not drawn from the training colours {colours}")
        if pair and not role_agnostic:
            raise ValueError("a pinned pair needs role_agnostic: nothing else names the objects in the prompt")
        self.pair = pair  # when set, every described episode asks for this pick on this place
        if exclude_real is None:
            exclude_real = roles.HELD_OUT_REAL if role_agnostic else ()
        skipped = (*TASK_LABELS, *exclude_real)
        self.real_objects = [o for o in real_objects or [] if o.label not in skipped]
        self.camera_pose, self.camera_mix = camera_pose, camera_mix
        self.camera_counts: Counter[str] = Counter()  # how many episodes came from each source
        self.last_camera_source = ""
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
            floor_texture=self.floor_texture(),
            wrist_camera=self.wrist_camera(),
        )

    def floor_texture(self) -> str:
        """A procedural floor for `texture_share` of episodes; draws nothing when the share is 0."""
        r = self.ranges
        if not r.texture_share or self.rng.random() >= r.texture_share:
            return ""
        return f"{self.rng.choice(KINDS)}:{int(self.rng.integers(*r.texture_seeds))}"

    def wrist_camera(self) -> dict[str, Any] | None:
        """The menagerie or Seeed wrist mount, jittered in position, tilt and field of view; None when not randomised."""
        r = self.ranges
        if not r.wrist_randomize:
            return None
        mount = SEEED_WRIST if self.rng.random() < r.seeed_share else MENAGERIE_WRIST
        return jittered_mount(mount, self.rng, r.wrist_jitter_m, r.wrist_tilt_jitter_deg, r.wrist_fovy_jitter_deg)

    def camera(self) -> dict[str, list[float] | float]:
        """This episode's camera: from the mix when a pose was given, else the wide arc. Counts each source."""
        source = "arc" if self.camera_pose is None else CAMERA_SOURCES[int(self.rng.choice(3, p=self.camera_mix))]
        self.camera_counts[source] += 1
        self.last_camera_source = source
        if source == "pose":
            return self.pose_camera()
        return self.layout.top_down_camera_spec() if source == "top_down" else self.arc_camera()

    def pose_camera(self) -> dict[str, list[float] | float]:
        """The calibrated pose with its position, orientation (yaw, pitch, roll) and field of view jittered."""
        r, pose = self.ranges, self.camera_pose
        assert pose is not None, "pose_camera needs a calibrated camera_pose"
        forward, up = np.array(pose.forward), np.array(pose.up)
        right = np.cross(forward, up)
        yaw, pitch, roll = (self.uniform((-r.pose_rotation_deg, r.pose_rotation_deg)) for _ in range(3))
        forward = rotated(forward, up, yaw)  # yaw about the camera's up axis
        forward, up = rotated(forward, right, pitch), rotated(up, right, pitch)  # pitch about its right axis
        up = rotated(up, forward, roll)  # roll about the optical axis
        shift = [self.uniform((-r.pose_position_mm, r.pose_position_mm)) for _ in range(3)]
        position = tuple(float(c + d) for c, d in zip(pose.position_mm, shift))
        jittered = replace(
            pose,
            position_mm=position,
            forward=tuple(float(v) for v in forward),
            up=tuple(float(v) for v in up),
            hfov_deg=pose.hfov_deg + self.uniform((-r.pose_hfov_deg, r.pose_hfov_deg)),
        )
        return self.layout.camera_spec(jittered)

    def arc_camera(self) -> dict[str, list[float] | float]:
        """A camera on an arc around the mat, aimed near its centre, image up = world up-ish."""
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
        """The named colour with hue, saturation and brightness jittered, relative to the mat's paper."""
        h, s, v = colorsys.rgb_to_hsv(*roles.PALETTE.get(label, (0.5, 0.5, 0.5)))
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
        aspect: tuple[float, float] = (1.0, 1.3),
        colour: str | None = None,
        shape: str | None = None,
    ) -> DetectedObject:
        """A random box or cylinder (or the given `shape`) named `label`, painted `colour` (default: the label's own)."""
        width = self.uniform(width_mm)
        cylinder = self.rng.random() < 0.5 if shape is None else shape == "cylinder"
        length = width if cylinder else min(width * self.uniform(aspect), max(max_length_mm, width))
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
            rgb=self.colour(label if colour is None else colour),
        )

    def objects(self, sampler: SceneSampler) -> list[DetectedObject]:
        """This episode's objects; see `episode` for the task that goes with them."""
        return self.episode(sampler).objects

    def episode(self, sampler: SceneSampler, task: str = STACK_TASK) -> Episode:
        """The next demo scene and its task: `blue` onto `red` with `task` as the prompt, or, when role-agnostic and
        not one of the `original_share`, two objects named by their looks with a prompt that names them."""
        if self.role_agnostic and self.rng.random() >= self.ranges.original_share:
            return self.described(sampler)
        return Episode(self.standard_objects(sampler), *TASK_LABELS, task)

    def standard_objects(self, sampler: SceneSampler) -> list[DetectedObject]:
        """The sampler's pick (blue) and place (red) positions as random shapes, plus distractors clear of the arm."""
        pick, place = sampler.sample()
        r = self.ranges
        found = [
            self.block("blue", r.pick_width_mm, r.pick_height_mm, r.pick_length_mm, (pick.x_mm, pick.y_mm)),
            self.block("red", r.place_width_mm, r.place_height_mm, xy=(place.x_mm, place.y_mm)),
        ]
        self.add_lookalikes(found, DISTRACTOR_LABELS, named=False)
        count = r.wide_distractors if self.role_agnostic else r.distractors
        for _ in range(self.count(count)):
            self.place_clear(found, self.distractor())
        return found

    def real_copy(self, real: DetectedObject) -> DetectedObject:
        """The real object with its size scaled by up to +-10% and a random yaw."""
        s = self.rng.uniform(0.9, 1.1)
        return replace(
            real,
            area_mm2=round(real.area_mm2 * s * s, 1),
            length_mm=round(real.length_mm * s, 1),
            width_mm=round(real.width_mm * s, 1),
            height_mm=round(real.height_mm * s, 1),
            yaw_deg=round(self.rng.uniform(-90, 90), 1),
        )

    def distractor(self) -> DetectedObject:
        """A copy of a real object half the time when there are any, else a random shape (larger ones when
        role-agnostic, to cover the real objects' sizes)."""
        if self.real_objects and self.rng.random() < REAL_DISTRACTOR_SHARE:
            return self.real_copy(self.real_objects[int(self.rng.integers(len(self.real_objects)))])
        label = str(self.rng.choice(DISTRACTOR_LABELS))
        if self.role_agnostic:
            return self.wide_block(label)
        return self.block(label, self.ranges.place_width_mm, self.ranges.distractor_height_mm)

    def count(self, bounds: tuple[int, int]) -> int:
        return int(self.rng.integers(bounds[0], bounds[1] + 1))

    def add_lookalikes(self, found: list[DetectedObject], colours: Sequence[str], named: bool) -> None:
        """Pick-sized decoys shaped like `found[0]` in `colours`. Draws nothing when the range is (0, 0), so
        recordings without lookalikes keep their scenes for a seed."""
        if not (n := self.count(self.ranges.lookalikes)):
            return
        r = self.ranges
        for colour in map(str, self.rng.permutation(colours)[:n]):
            decoy = self.block(colour, r.pick_width_mm, r.pick_height_mm, r.pick_length_mm, shape=found[0].shape)
            self.place_clear(found, self.named(decoy, colour) if named else decoy)

    def wide_block(self, colour: str) -> DetectedObject:
        r = self.ranges
        return self.block(
            colour,
            r.wide_distractor_width_mm,
            r.wide_distractor_height_mm,
            MAX_DISTRACTOR_LENGTH_MM,
            aspect=r.wide_distractor_aspect,
        )

    def named(self, obj: DetectedObject, colour: str) -> DetectedObject:
        """The object labelled by what it looks like: its colour and a noun that fits its footprint (`green_can`)."""
        noun = str(self.rng.choice(roles.nouns_for(obj.shape, obj.length_mm, obj.width_mm)))
        return replace(obj, label=f"{colour}_{noun}")

    def described(self, sampler: SceneSampler, place_on: DetectedObject | None = None) -> Episode:
        """Pick and place objects named by their looks, the place target a real object in a few episodes (always
        `place_on`, when given), and clutter that never shares a colour (or a name) with them, so the prompt picks
        out exactly one object each. Pick objects are always synthetic: the real ones are wider than the jaws open."""
        pick_at, place_at = sampler.sample()
        r = self.ranges
        if self.pair:  # a pinned pair draws nothing: its episodes vary in all but what is asked for
            pick_colour, place_colour = self.pair
        else:
            pick_colour, place_colour = (str(c) for c in self.rng.choice(self.colours, 2, replace=False))
        pick = self.block(
            pick_colour, r.pick_width_mm, r.pick_height_mm, r.pick_length_mm, (pick_at.x_mm, pick_at.y_mm)
        )
        found = [self.named(pick, pick_colour)]
        if place_on is not None or (self.real_objects and self.rng.random() < r.real_place_share):
            real = place_on or self.real_objects[int(self.rng.integers(len(self.real_objects)))]
            found.append(replace(self.real_copy(real), x_mm=place_at.x_mm, y_mm=place_at.y_mm))
        else:
            place = self.block(place_colour, r.place_width_mm, r.place_height_mm, xy=(place_at.x_mm, place_at.y_mm))
            found.append(self.named(place, place_colour))
        self.add_lookalikes(found, [c for c in self.colours if c not in (pick_colour, place_colour)], named=True)
        for _ in range(self.count(r.wide_distractors)):
            if (candidate := self.described_distractor(found)) is not None:
                self.place_clear(found, candidate)
        return Episode(found, found[0].label, found[1].label, roles.instruction(found[0].label, found[1].label))

    def described_distractor(self, taken: list[DetectedObject]) -> DetectedObject | None:
        """A real object not yet in the scene, or a block in a colour nothing else has; None when there is neither."""
        names = {o.label for o in taken}
        spare = [o for o in self.real_objects if o.label not in names]
        if spare and self.rng.random() < REAL_DISTRACTOR_SHARE:
            return self.real_copy(spare[int(self.rng.integers(len(spare)))])
        used = {o.label.split("_")[0] for o in taken}
        if not (free := [c for c in roles.TRAIN_COLOURS if c not in used]):
            return None
        colour = str(self.rng.choice(free))
        return self.named(self.wide_block(colour), colour)

    def place_clear(self, taken: list[DetectedObject], candidate: DetectedObject) -> None:
        """Add the candidate at a `free_spot`, or leave it out when there is none."""
        if (xy := self.free_spot(taken, candidate)) is not None:
            taken.append(replace(candidate, x_mm=round(float(xy[0]), 1), y_mm=round(float(xy[1]), 1)))

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
