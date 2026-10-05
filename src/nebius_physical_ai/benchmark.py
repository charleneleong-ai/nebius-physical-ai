"""The benchmark every checkpoint is scored on: fixed scenes, the same success check, seed 42.

The ladder is measure, adapt, verify, promote; this is the measuring stick. A *suite* is a list of *cases* (a scene,
the task, the camera) and a number of randomised rehearsal variants per case; its score is successes over all variants
with a Wilson interval. Suites come in two groups:

- `legacy`: the cells the v2 and v3 numbers were taken from (the blue and red cubes from a 65 degree camera, real
  objects as clutter, an azimuth sweep), rebuilt exactly, so a new checkpoint can be put next to the old ones.
- `held_out`: role-agnostic prompts on objects the training randomiser never produces (colours, sizes and real objects
  in `twin.roles` and `HELD_OUT_RANGES`), and unseen scenes of the trained vocabulary for contrast.

Generated scenes are kept only if the scripted expert can solve them, so a miss is the policy's, not the scene's.
Success is judged on the *named* pair of blocks, so stacking the wrong object counts as a failure.
"""

import fnmatch
import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler, UnreachableTarget, start_pose
from nebius_physical_ai.twin.randomize import Randomization, SceneRandomizer
from nebius_physical_ai.twin.rehearsal import Rehearsal, RehearsalResult
from nebius_physical_ai.twin.roles import HELD_OUT_COLOURS, HELD_OUT_REAL, TRAIN_COLOURS
from nebius_physical_ai.twin.rollout import STACK_TASK, StackCheck
from nebius_physical_ai.twin.scene import SceneLook, Twin, TwinLayout

SEED = 42
HEADLINE_N = 96  # rehearsals per cell: the same scene moves by about 8 of 48 between runs, so smaller cells mislead
OBLIQUE_ELEVATION_DEG = 65.0
AZIMUTHS_DEG = (0.0, 90.0, 130.0, 160.0, 180.0)  # 0 = across the mat facing the robot, 180 = behind the arm
GENERATED_CASES = 12
MAX_TRIES_PER_CASE = 20
CLEAN_BLUE = DetectedObject("blue", 131.0, 151.0, 900.0)
CLEAN_RED = DetectedObject("red", 221.0, 151.0, 1600.0)
# Sizes the training randomiser never draws: taller blocks to carry, wider and taller place targets.
# Real place targets are the `place_on_*` suites' business, so here both roles are synthetic.
HELD_OUT_RANGES = Randomization(
    pick_height_mm=(41.0, 50.0),
    place_height_mm=(34.0, 42.0),
    place_width_mm=(62.0, 76.0),
    wide_distractors=(0, 2),
    real_place_share=0.0,
)
SEEN_RANGES = Randomization(wide_distractors=(0, 2), real_place_share=0.0)


@dataclass(frozen=True)
class FixedCameraLayout(TwinLayout):
    """A layout whose front camera is one given MuJoCo camera spec (None = the layout's own)."""

    camera: dict[str, Any] | None = None

    def front_camera_spec(self) -> dict[str, Any]:
        return self.camera if self.camera is not None else super().front_camera_spec()


@dataclass(frozen=True)
class Case:
    """One scene to rehearse: its objects, the task in the policy's words and in block names, and the camera."""

    name: str
    objects: list[DetectedObject]
    pick: str
    place: str
    prompt: str
    camera: dict[str, Any] | None = None  # None = straight down

    def layout(self, base: TwinLayout) -> FixedCameraLayout:
        return FixedCameraLayout(base.mat, base.base_offset_mm, base.front_camera, self.camera)


@dataclass(frozen=True)
class Suite:
    name: str
    group: str
    cases: list[Case]
    note: str = ""
    rejected: int = 0  # generated scenes the expert could not solve, left out

    @property
    def variants(self) -> int:
        return math.ceil(HEADLINE_N / len(self.cases)) if self.cases else 0

    @property
    def n(self) -> int:
        return self.variants * len(self.cases)


@dataclass
class Cell:
    """A suite's score for one checkpoint."""

    suite: str
    group: str
    successes: int
    n: int
    cases: int
    seconds: float
    per_case: list[int] = field(default_factory=list)  # successes per case, in suite order
    # First success and first failure rollout, kind -> (caption, frames); only when videos were asked for, never saved
    examples: dict[str, tuple[str, list[np.ndarray]]] = field(default_factory=dict, repr=False, compare=False)

    @property
    def p(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def wilson(self) -> tuple[float, float]:
        return RehearsalResult(self.successes, self.n, [], []).wilson() if self.n else (0.0, 1.0)

    def record(self) -> dict[str, Any]:
        """JSON-ready row: the counts, the rate and its interval (not the rollout frames)."""
        lo, hi = self.wilson()
        saved = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "examples"}
        return saved | {"p": round(self.p, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def oblique_camera(layout: TwinLayout, elevation_deg: float, azimuth_deg: float) -> dict[str, Any]:
    """A camera 500 mm from the mat's centre at the given elevation and azimuth (the one the sweeps used)."""
    ranges = Randomization(
        elevation_deg=(elevation_deg,) * 2,
        azimuth_deg=(azimuth_deg,) * 2,
        distance_mm=(500.0,) * 2,
        fovy_deg=(42.0,) * 2,
        aim_jitter_mm=0.0,
    )
    return SceneRandomizer(layout, seed=SEED, ranges=ranges).arc_camera()


def clean_case(name: str, camera: dict[str, Any] | None) -> Case:
    return Case(name, [CLEAN_BLUE, CLEAN_RED], "blue", "red", STACK_TASK, camera)


def clutter_scenes(real: list[DetectedObject], layout: TwinLayout) -> dict[str, list[DetectedObject]]:
    """The clean pair with real objects placed clear of the arm (one, two, all), and one in the arm's path."""
    placer = SceneRandomizer(layout, seed=SEED, real_objects=real)
    pair = [CLEAN_BLUE, CLEAN_RED]
    clear = list(pair)
    for obj in real:
        if (spot := placer.free_spot(clear, obj)) is not None:
            clear.append(replace(obj, x_mm=round(float(spot[0]), 1), y_mm=round(float(spot[1]), 1)))
    placed = {o.label: o for o in clear[2:]}
    scenes = {f"real_1_{label}": [*pair, placed[label]] for label in placed}
    scenes |= {f"real_2_{a}+{b}": [*pair, placed[a], placed[b]] for a, b in itertools.combinations(placed, 2)}
    if len(placed) > 2:
        scenes[f"real_{len(placed)}"] = [*pair, *placed.values()]
    if real:
        base = np.array([layout.mat.width_mm / 2, layout.mat.height_mm + layout.base_offset_mm])
        mid = (base + np.array([CLEAN_BLUE.x_mm, CLEAN_BLUE.y_mm])) / 2  # on the arm's path to the blue cube
        scenes[f"in_path_{real[0].label}"] = [
            *pair,
            replace(real[0], x_mm=round(float(mid[0]), 1), y_mm=round(float(mid[1]), 1)),
        ]
    return scenes


def legacy_suites(real: list[DetectedObject], layout: TwinLayout, calibrated: CameraPose | None = None) -> list[Suite]:
    """Rebuilds the v2 and v3 evaluation cells."""
    oblique = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, 0.0)
    suites = [Suite("clean", "legacy", [clean_case("clean", oblique)], "blue onto red from 65 degrees, facing the arm")]
    suites.append(Suite("clean_top_down", "legacy", [clean_case("clean_top_down", None)], "the v1 and v2 camera"))
    for azimuth in AZIMUTHS_DEG[1:]:
        camera = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, azimuth)
        suites.append(Suite(f"azimuth_{azimuth:.0f}", "legacy", [clean_case(f"azimuth_{azimuth:.0f}", camera)]))
    if calibrated is not None:
        camera = layout.camera_spec(calibrated)
        suites.append(Suite("clean_calibrated", "legacy", [clean_case("clean_calibrated", camera)], "the real camera"))
    for name, objects in clutter_scenes(real, layout).items():
        case = Case(name, objects, "blue", "red", STACK_TASK, oblique)
        suites.append(
            Suite(name, "legacy", [case], "real objects as clutter, seen or not by training depends on v-number")
        )
    return suites


def expert_solves(case: Case, layout: TwinLayout, start: np.ndarray) -> bool:
    """Whether the scripted expert completes this case: if it can't, a policy miss says nothing about the policy."""
    own = case.layout(layout)
    with Twin.from_objects(case.objects, own, start_pose=start, look=SceneLook(camera=case.camera)) as twin:
        try:
            list(PickPlaceExpert(twin, case.prompt, case.pick, case.place, image_size=(24, 32)).run())
        except UnreachableTarget:
            return False
        twin.settle(0.5)
        return StackCheck(case.pick, case.place)(twin)


def solvable_cases(
    name: str,
    draw: Callable[[], tuple[list[DetectedObject], str, str, str]],
    layout: TwinLayout,
    camera: dict[str, Any],
    count: int = GENERATED_CASES,
) -> tuple[list[Case], int]:
    """Up to `count` cases from `draw` that the expert solves, and how many draws it did not."""
    start, cases, rejected = start_pose(np.random.default_rng(SEED)), [], 0
    for _ in range(count * MAX_TRIES_PER_CASE):
        objects, pick, place, prompt = draw()
        case = Case(f"{name}_{len(cases)}", objects, pick, place, prompt, camera)
        if expert_solves(case, layout, start):
            cases.append(case)
        else:
            rejected += 1
        if len(cases) == count:
            break
    return cases, rejected


def held_out_suites(real: list[DetectedObject], layout: TwinLayout) -> list[Suite]:
    """Role-agnostic cases on what the training randomiser never produces, plus unseen scenes of what it does."""
    camera = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, 0.0)
    reals = {o.label: o for o in real}

    def episode_draw(randomizer: SceneRandomizer, sampler: SceneSampler, place_on: DetectedObject | None = None):
        def draw() -> tuple[list[DetectedObject], str, str, str]:
            e = randomizer.described(sampler, place_on)
            return e.objects, e.pick, e.place, e.prompt

        return draw

    def build(
        name: str, note: str, seed: int, colours: tuple[str, ...], ranges: Randomization, place_on: str | None = None
    ) -> Suite:
        # Benchmark scenes come from their own random streams, apart from the demos' (seed 42 itself).
        randomizer = SceneRandomizer(
            layout,
            seed=[SEED, seed],
            ranges=ranges,
            real_objects=real,
            role_agnostic=True,
            colours=colours,
            exclude_real=(),
        )
        sampler = SceneSampler(layout, seed=[SEED, seed])
        draw = episode_draw(randomizer, sampler, reals.get(place_on) if place_on else None)
        cases, rejected = solvable_cases(name, draw, layout, camera)
        return Suite(name, "held_out", cases, note, rejected)

    suites = [
        build("held_out_objects", "colours and sizes outside training", 1, HELD_OUT_COLOURS, HELD_OUT_RANGES),
        build("seen_vocabulary", "new scenes, trained colours and sizes", 2, TRAIN_COLOURS, SEEN_RANGES),
    ]
    for i, label in enumerate((*HELD_OUT_REAL, "pomade_tin")):
        if label in reals:
            note = "place target never seen in training" if label in HELD_OUT_REAL else "place target seen in training"
            suites.append(build(f"place_on_{label}", note, 10 + i, TRAIN_COLOURS, SEEN_RANGES, label))
    return suites


def diagnostic_suites(layout: TwinLayout) -> list[Suite]:
    """`clean_many`: the original task (blue onto red, the original prompt) in many random scenes from the 65 degree
    camera. The legacy `clean` cell is one scene, so on its own it cannot say whether a drop is general forgetting or
    one bad scene; this suite can."""
    camera = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, 0.0)
    randomizer = SceneRandomizer(layout, seed=[SEED, 20])
    sampler = SceneSampler(layout, seed=[SEED, 20])

    def draw() -> tuple[list[DetectedObject], str, str, str]:
        return randomizer.standard_objects(sampler), "blue", "red", STACK_TASK

    cases, rejected = solvable_cases("clean_many", draw, layout, camera)
    note = "the original task in 12 random scenes from the clean camera"
    return [Suite("clean_many", "diagnostic", cases, note, rejected)]


def build_suites(
    real: list[DetectedObject],
    layout: TwinLayout,
    group: str = "all",
    only: Iterable[str] = (),
    calibrated: CameraPose | None = None,
) -> list[Suite]:
    """The suites of `group` (legacy, held_out, diagnostic or all) whose names match any glob in `only`."""
    suites: list[Suite] = []
    if group in ("legacy", "all"):
        suites += legacy_suites(real, layout, calibrated)
    if group in ("held_out", "all"):
        suites += held_out_suites(real, layout)
    if group in ("diagnostic", "all"):
        suites += diagnostic_suites(layout)
    patterns = list(only)
    return [s for s in suites if not patterns or any(fnmatch.fnmatch(s.name, p) for p in patterns)]


def run_suite(
    suite: Suite, policy: Policy, layout: TwinLayout, batch_size: int = 32, record_videos: bool = False
) -> Cell:
    """Rehearse every case of the suite `variants` times and pool the successes.

    With `record_videos` the cell also keeps the first success and the first failure as rollout frames."""
    import time

    start, per_case = time.time(), []
    examples: dict[str, tuple[str, list[np.ndarray]]] = {}
    for case in suite.cases:
        rehearsal = Rehearsal(
            policy,
            case.prompt,
            StackCheck(case.pick, case.place),
            case.layout(layout),
            batch_size=batch_size,
            record_examples=record_videos,
        )
        result = rehearsal.run(case.objects, suite.variants, SEED)
        per_case.append(result.successes)
        for kind, (index, frames) in result.examples.items():
            caption = f"{suite.name} | {case.name} | {kind} | variant {index} | {case.prompt}"
            examples.setdefault(kind, (caption, frames))
    cell = Cell(
        suite.name, suite.group, sum(per_case), suite.n, len(suite.cases), round(time.time() - start, 1), per_case
    )
    cell.examples = examples
    return cell


def verdict(a: Cell, b: Cell) -> str:
    """How `a` compares with `b`: better or worse only when their 95% intervals don't overlap, else within noise."""
    (a_lo, a_hi), (b_lo, b_hi) = a.wilson(), b.wilson()
    if a_lo > b_hi:
        return "better"
    return "worse" if a_hi < b_lo else "within noise"
