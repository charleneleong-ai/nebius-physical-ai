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
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler, UnreachableTarget, start_pose
from nebius_physical_ai.twin.randomize import SEEED_WRIST, Randomization, SceneRandomizer
from nebius_physical_ai.twin.rehearsal import Rehearsal, RehearsalResult
from nebius_physical_ai.twin.roles import HELD_OUT_COLOURS, HELD_OUT_REAL, TRAIN_COLOURS
from nebius_physical_ai.twin.rollout import STACK_TASK, Check, StackCheck
from nebius_physical_ai.twin.scene import Fixture, SceneLook, Twin, TwinLayout
from nebius_physical_ai.twin.tasks import Basket, BasketCheck, Peg, PegCheck, Ring, TaskCheck, TaskExpert
from nebius_physical_ai.twin.textures import HELD_OUT_SEED, KINDS

SEED = 42
HEADLINE_N = 96  # rehearsals per cell: the same scene moves by about 8 of 48 between runs, so smaller cells mislead
HEADLINE_SUITES = (  # the promotion gate's six suites (also listed in scripts/dagger_round.sh)
    "held_out_objects",
    "seen_vocabulary",
    "clean_many",
    "clean_top_down_many",
    "real_1_pomade_tin_many",
    "in_path_pomade_tin_many",
)
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
    look: SceneLook | None = None  # floor texture and wrist camera; None = the plain twin
    fixtures: tuple[Fixture, ...] = ()  # `twin.tasks` fixtures (a basket, a peg, a ring) beside the blocks
    check: TaskCheck | None = None  # the task's success test; None = stack `pick` on `place`

    def layout(self, base: TwinLayout) -> FixedCameraLayout:
        return FixedCameraLayout(base.mat, base.base_offset_mm, base.front_camera, self.camera)

    def success(self) -> Check:
        return self.check or StackCheck(self.pick, self.place)


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
    outcomes: list[bool] = field(default_factory=list)  # every rehearsal, case-major: pairs checkpoints scene by scene
    failures: dict[str, int] = field(default_factory=dict)  # how the failed rehearsals failed (FailureWatch)
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
    look = SceneLook(camera=case.camera)
    with Twin.from_objects(case.objects, own, start_pose=start, look=look, fixtures=case.fixtures) as twin:
        expert = (
            TaskExpert(twin, case.prompt, case.check, image_size=(24, 32))
            if case.check
            else PickPlaceExpert(twin, case.prompt, case.pick, case.place, image_size=(24, 32))
        )
        try:
            list(expert.run())
        except UnreachableTarget:
            return False
        twin.settle(0.5)
        return case.success()(twin)


def solvable_cases(
    name: str,
    draw: Callable[[], Case],
    layout: TwinLayout,
    camera: dict[str, Any] | None,
    count: int = GENERATED_CASES,
) -> tuple[list[Case], int]:
    """Up to `count` cases from `draw` that the expert solves, and how many draws it did not."""
    start, cases, rejected = start_pose(np.random.default_rng(SEED)), [], 0
    for _ in range(count * MAX_TRIES_PER_CASE):
        case = replace(draw(), name=f"{name}_{len(cases)}")
        if expert_solves(case, layout, start):
            cases.append(case)
        else:
            rejected += 1
        if len(cases) == count:
            break
    return cases, rejected


EYE_DROPS_MM = (26.0, 40.0)  # diameter, height: narrower than ~24 mm slips out of the closed gripper
TASK_GAP_MM = 120.0  # object to fixture, so the gripper never starts inside the basket or the ring over the peg
DECOY_SIDE_MM = 30.0
DECOY_GAP_MM = 48.0 + DECOY_SIDE_MM / 2 + 15.0  # the ring's outer radius plus the decoy's half side, plus margin


def held_out_task_suites(layout: TwinLayout) -> list[Suite]:
    """Tasks no checkpoint here was trained on, after ArmnetBench's eye_drops_to_basket and ring_insert, in its own
    words: an object and a fixture placed apart in the arm's reach, plus one training-colour cube as a decoy. Variants
    perturb the eye drops and the decoy; the basket, the peg and the ring stay put. Every case is one the scripted
    expert solves (`TaskExpert`); the ring is grasped on the side along the reach, the only grasp it holds."""
    camera = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, 0.0)

    sampler = SceneSampler(layout)

    def spots(rng: np.random.Generator) -> np.ndarray:
        """Object, fixture and decoy positions (mat mm) inside reach: object and fixture TASK_GAP_MM apart."""
        while True:
            xy = rng.uniform([sampler.x_range[0], sampler.y_range[0]], [sampler.x_range[1], sampler.y_range[1]], (3, 2))
            gaps = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
            if gaps[0, 1] >= TASK_GAP_MM and min(gaps[2, 0], gaps[2, 1]) >= DECOY_GAP_MM:
                return xy

    def decoy(rng: np.random.Generator, xy: np.ndarray) -> DetectedObject:
        return DetectedObject(str(rng.choice(TRAIN_COLOURS)), *np.round(xy, 1), DECOY_SIDE_MM**2)

    def basket_draw(rng: np.random.Generator) -> Callable[[], Case]:
        def draw() -> Case:
            (drops_xy, basket_xy, decoy_xy) = spots(rng)
            dia, height = EYE_DROPS_MM
            drops = DetectedObject("eye_drops", *np.round(drops_xy, 1), dia * dia, length_mm=dia, width_mm=dia,
                                   height_mm=height, shape="cylinder", rgb=(0.92, 0.92, 0.95))  # fmt: skip
            basket = Basket(*np.round(basket_xy, 1))
            return Case(name="", objects=[drops, decoy(rng, decoy_xy)], pick="eye_drops", place=basket.name,
                        prompt="Put the eye drops into the basket", camera=camera, fixtures=(basket,),
                        check=BasketCheck("eye_drops", basket))  # fmt: skip

        return draw

    def ring_draw(rng: np.random.Generator) -> Callable[[], Case]:
        def draw() -> Case:
            (ring_xy, peg_xy, decoy_xy) = spots(rng)
            ring, peg = Ring(*np.round(ring_xy, 1), radius_mm=36.0, tube_mm=12.0), Peg(*np.round(peg_xy, 1))
            check = PegCheck(ring, peg, float(rng.choice([0.0, np.pi])))
            return Case(name="", objects=[decoy(rng, decoy_xy)], pick=ring.name, place=peg.name,
                        prompt="Insert the colourful ring into the central wooden peg", camera=camera,
                        fixtures=(ring, peg), check=check)  # fmt: skip

        return draw

    suites = []
    for name, stream, draw, note in (
        ("held_out_task_basket", 30, basket_draw, "eye drops into a basket (ArmnetBench eye_drops_to_basket)"),
        ("held_out_task_ring", 31, ring_draw, "ring over a peg (ArmnetBench ring_insert)"),
    ):
        cases, rejected = solvable_cases(name, draw(np.random.default_rng([SEED, stream])), layout, camera)
        suites.append(Suite(name, "held_out_tasks", cases, note, rejected))
    return suites


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
        cases, rejected = solvable_cases(name, lambda: Case("", *draw(), camera=camera), layout, camera)
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


def diagnostic_suites(layout: TwinLayout, real: list[DetectedObject] | None = None) -> list[Suite]:
    """Many-scene versions of the single-scene legacy cells, so a drop can be told apart from one bad scene.

    `clean_many` is the original task (blue onto red, the original prompt) in 12 random scenes from the 65 degree camera;
    `clean_top_down_many` the same from the v1/v2 camera. With a real object, `real_1_<name>_many` adds it clear of the
    arm and `in_path_<name>_many` halfway along the arm's path to the blue block, both to just the pair (as the legacy
    cells do). Each suite draws from its own seed stream."""
    oblique = oblique_camera(layout, OBLIQUE_ELEVATION_DEG, 0.0)
    base = np.array([layout.mat.width_mm / 2, layout.mat.height_mm + layout.base_offset_mm])
    tin = next((o for o in real or [] if o.label == "pomade_tin"), (real or [None])[0])

    def suite(name: str, stream: int, camera: dict[str, Any] | None, add: Callable[..., list[DetectedObject]]) -> Suite:
        randomizer = SceneRandomizer(layout, seed=[SEED, stream])
        sampler = SceneSampler(layout, seed=[SEED, stream])

        def draw() -> tuple[list[DetectedObject], str, str, str]:
            return add(randomizer, randomizer.standard_objects(sampler)), "blue", "red", STACK_TASK

        cases, rejected = solvable_cases(name, lambda: Case("", *draw(), camera=camera), layout, camera)
        return Suite(name, "diagnostic", cases, f"{name.removesuffix('_many')} in 12 random scenes", rejected)

    def keep(_: SceneRandomizer, objects: list[DetectedObject]) -> list[DetectedObject]:
        return objects

    def clear_of_arm(randomizer: SceneRandomizer, objects: list[DetectedObject]) -> list[DetectedObject]:
        pair = objects[:2]
        spot = randomizer.free_spot(pair, tin)
        return (
            pair
            if spot is None
            else [*pair, replace(tin, x_mm=round(float(spot[0]), 1), y_mm=round(float(spot[1]), 1))]
        )

    def in_path(_: SceneRandomizer, objects: list[DetectedObject]) -> list[DetectedObject]:
        blue = objects[0]
        mid = (base + np.array([blue.x_mm, blue.y_mm])) / 2
        return [*objects[:2], replace(tin, x_mm=round(float(mid[0]), 1), y_mm=round(float(mid[1]), 1))]

    suites = [suite("clean_many", 20, oblique, keep), suite("clean_top_down_many", 21, None, keep)]
    if tin is not None:
        suites += [
            suite(f"real_1_{tin.label}_many", 22, oblique, clear_of_arm),
            suite(f"in_path_{tin.label}_many", 23, oblique, in_path),
        ]
    return suites


def appearance_suites(layout: TwinLayout) -> list[Suite]:
    """`clean_many`'s scenes seen differently, so a checkpoint's score on them against `clean_many` (paired, scene by
    scene) isolates what appearance alone costs: on floors textured from held-out seeds (wood, fabric and checker in
    turn), and through the Seeed wrist mount so101-benchmark uses instead of the menagerie one."""
    clean = next(s for s in diagnostic_suites(layout) if s.name == "clean_many")

    def seen(name: str, look: Callable[[int], SceneLook], note: str) -> Suite:
        cases = [replace(c, name=f"{name}_{i}", look=look(i)) for i, c in enumerate(clean.cases)]
        return Suite(name, "appearance", cases, note, clean.rejected)

    def textured(i: int) -> SceneLook:
        return SceneLook(floor_texture=f"{KINDS[i % len(KINDS)]}:{HELD_OUT_SEED + i}")

    def seeed(_: int) -> SceneLook:
        return SceneLook(wrist_camera=SEEED_WRIST)

    return [
        seen("clean_many_textured", textured, "clean_many on held-out procedural floors"),
        seen("clean_many_seeed_wrist", seeed, "clean_many through the Seeed wrist mount"),
    ]


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
        suites += diagnostic_suites(layout, real)
    if group in ("appearance", "all"):
        suites += appearance_suites(layout)
    if group == "held_out_tasks":  # not in "all": a new task, scored apart from the stacking gate
        suites += held_out_task_suites(layout)
    patterns = list(only)
    return [s for s in suites if not patterns or any(fnmatch.fnmatch(s.name, p) for p in patterns)]


def run_suite(
    suite: Suite,
    policy: Policy,
    layout: TwinLayout,
    batch_size: int = 32,
    record_videos: bool = False,
    on_case: Callable[[str, int, int], None] | None = None,
) -> Cell:
    """Rehearse every case of the suite `variants` times and pool the successes.

    With `record_videos` the cell also keeps the first success and the first failure as rollout frames. `on_case` is
    called after each case with its name, the suite's successes so far and the rehearsals so far."""
    import time

    start, per_case, outcomes, failures = time.time(), [], [], Counter()
    examples: dict[str, tuple[str, list[np.ndarray]]] = {}
    for case in suite.cases:
        rehearsal = Rehearsal(
            policy,
            case.prompt,
            case.success(),
            case.layout(layout),
            batch_size=batch_size,
            look=case.look,
            record_examples=record_videos,
            fixtures=case.fixtures,
        )
        result = rehearsal.run(case.objects, suite.variants, SEED)
        per_case.append(result.successes)
        outcomes += [bool(v["success"]) for v in result.variants]
        if on_case:
            on_case(case.name, sum(per_case), len(outcomes))
        failures.update(v["failure"] for v in result.variants if not v["success"])
        for kind, (index, frames) in result.examples.items():
            caption = f"{suite.name} | {case.name} | {kind} | variant {index} | {case.prompt}"
            examples.setdefault(kind, (caption, frames))
    cell = Cell(
        suite.name,
        suite.group,
        sum(per_case),
        suite.n,
        len(suite.cases),
        round(time.time() - start, 1),
        per_case,
        outcomes,
        dict(failures.most_common()),
    )
    cell.examples = examples
    return cell


def mcnemar(a: Cell, b: Cell) -> tuple[int, int, float] | None:
    """Exact McNemar test on the scenes both cells rehearsed: (a wins, b wins, two-sided p), or None if unpaired.

    Every checkpoint sees the same scenes at the same seed, so only the scenes where they disagree carry evidence; this
    is far more sensitive than comparing two independent intervals."""
    if not a.outcomes or len(a.outcomes) != len(b.outcomes):
        return None
    wins = sum(x and not y for x, y in zip(a.outcomes, b.outcomes))
    losses = sum(y and not x for x, y in zip(a.outcomes, b.outcomes))
    k, n = min(wins, losses), wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return wins, losses, p


def promotion(new: dict[str, Cell], ref: dict[str, Cell]) -> tuple[bool, str]:
    """Whether `new` should replace `ref`: better on every rehearsal pooled across their shared suites (paired), and
    worse on none of them. A round that only moves within noise is not promoted, so the served policy never drifts."""
    shared = [suite for suite in ref if suite in new]
    if not shared:
        return False, "no suites in common"
    if worse := [suite for suite in shared if verdict(new[suite], ref[suite]) == "worse"]:
        return False, f"worse on {', '.join(worse)}"
    pool = [
        Cell("pooled", "pooled", sum(c.successes for c in cells), sum(c.n for c in cells), 0, 0.0,
             outcomes=[o for c in cells for o in c.outcomes])
        for cells in ([new[s] for s in shared], [ref[s] for s in shared])
    ]  # fmt: skip
    pooled = verdict(*pool)
    gained = f"{pool[0].successes - pool[1].successes:+d} rehearsals over {len(shared)} suites, pooled {pooled}"
    return pooled == "better", gained


def verdict(a: Cell, b: Cell, alpha: float = 0.05) -> str:
    """How `a` compares with `b`. Paired (McNemar, p < alpha) when both have per-scene outcomes; otherwise better or
    worse only when their 95% intervals don't overlap. Anything else is within noise."""
    if (paired := mcnemar(a, b)) is not None:
        wins, losses, p = paired
        return "within noise" if p >= alpha else ("better" if wins > losses else "worse")
    (a_lo, a_hi), (b_lo, b_hi) = a.wilson(), b.wilson()
    if a_lo > b_hi:
        return "better"
    return "worse" if a_hi < b_lo else "within noise"
