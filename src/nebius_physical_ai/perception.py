"""Marker-corrected scene perception.

Geometry (positions, footprints, yaw, colour in mat millimetres) comes from the ArUco-rectified mat view. The vision
model (Cosmos 3, or the Token Factory fallback) supplies semantics only: labels, plausibility issues, and a
pixel-box cross-check that is projected through the same marker homography. Asking a vision model for millimetres
directly was off by 15-35 mm on a rendered scene, against the ~5 mm of perception noise the rehearsal assumes.
"""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from nebius_physical_ai.capture.mat import COLOUR_HUES, DetectedObject, MatLocation, SceneObservation
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec, matches, slug, words
from nebius_physical_ai.trust.verdict import Fidelity

# bbox_2d is normalised to 0-1000 on both axes, (x1, y1, x2, y2). Measured on Cosmos 3 Nano with scripts/verify_bbox.py:
# box centres land within 1.3 px on synthetic frames and 3-4 mm on real photos of the mat.
BOX_SCALE = 1000.0
MATCH_MM = 50.0  # a vision-model box within this of a blob (same label for task blocks) confirms it
UNCONFIRMED_CONFIDENCE = 0.5  # a task block the vision model did not confirm
# Marker fit residual (RMS mm) at which the markers' evidence counts for nothing. Measured on synthetic frames: a clean fit
# is ~0.3 mm and one marker printed 18 mm off its corner is only ~2 mm, because the homography absorbs most of it, so the
# tolerance is tight enough for that to cross the 0.6 ASK_HUMAN line. Tune on real phone frames.
MARKER_TOLERANCE_MM = 5.0
VL_MAX_EDGE_PX = 1280  # the vision model sees a downscaled frame; the markers are found at full resolution
# Words that mark a vision-model detection as part of the rig rather than an object on the mat. Left to itself the model
# lists the four printed markers (and their boxes project to the true corners within 2 mm).
RIG_WORDS = frozenset({"marker", "markers", "aruco", "fiducial", "qr", "arm", "robot", "gripper"})

SCENE_PROMPT = """Analyze this robot workspace image: a mat with four corner markers and objects on it.
Return JSON only:
{
  "objects": [{"label": "blue", "bbox_2d": [x1, y1, x2, y2], "shape": "box", "confidence": 0.9}],
  "relations": [{"type": "on|near|left_of|right_of", "subject": "...", "object": "..."}],
  "affordances": [{"object": "...", "grasp_type": "top|side"}],
  "issues": ["anything physically implausible or unsafe, e.g. a block already stacked or an object in the arm's path"]
}
List every object on the mat, including ones that are not brightly coloured, but not the printed corner markers or
the robot arm. Use the colour name as `label` for
coloured blocks (blue, red, yellow, green, purple) and a short noun otherwise. `bbox_2d` is the object's bounding box in
image coordinates normalised to 0-1000: (0, 0) is the image's top-left corner and (1000, 1000) its bottom-right.
`confidence` is in [0, 1]. `issues` is an empty list when the layout looks physically plausible."""


@dataclass
class CosmosSceneGraph:
    objects: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    affordances: list[dict[str, Any]]
    issues: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, content: dict[str, Any]) -> CosmosSceneGraph:
        issues = content.get("issues") or []
        return cls(
            objects=content.get("objects", []),
            relations=content.get("relations", []),
            affordances=content.get("affordances", []),
            issues=[str(i) for i in issues] if isinstance(issues, list) else [str(issues)],
        )


class SceneError(ValueError):
    """The marker view can't support a rehearsal (a task block is missing); the answer is ASK_HUMAN."""


@dataclass(frozen=True)
class VlDetection:
    label: str
    x_mm: float
    y_mm: float
    confidence: float


@dataclass
class MarkerScene:
    """What the twin is built from: [pick, place, *distractors] (blue and red by default), plus how far to trust it."""

    objects: list[DetectedObject]
    markers_found: list[int]
    residual_mm: float
    perception: float  # in [0, 1]: marker count and fit, times the task blocks' least confidence
    warnings: list[str] = field(default_factory=list)


def decode_frame(frame_b64: str) -> np.ndarray:
    """A base64 image (JPEG, PNG) to a BGR array; ValueError when it is not an image."""
    try:
        data = base64.b64decode(frame_b64, validate=True)
    except ValueError as err:
        raise ValueError("frame is not valid base64") from err
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("frame is not a decodable image")
    return image


def vl_jpeg_b64(image: np.ndarray, max_edge: int = VL_MAX_EDGE_PX) -> str:
    """The frame as a base64 JPEG no larger than `max_edge`, keeping the vision model's prompt a sane size."""
    scale = max_edge / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode the frame as JPEG")
    return base64.b64encode(jpeg.tobytes()).decode()


def vl_detections(
    graph: CosmosSceneGraph, location: MatLocation, frame_size: tuple[int, int]
) -> tuple[list[VlDetection], list[str]]:
    """Vision-model boxes (normalised 0-1000) projected through the marker homography to mat millimetres.

    `frame_size` is the (width, height) of the frame the markers were found in. Objects without a usable box are
    skipped with a note rather than guessed at.
    """
    width, height = frame_size
    detections, notes = [], []
    for raw in graph.objects:
        try:
            x1, y1, x2, y2 = (float(v) for v in raw["bbox_2d"])
            confidence = float(raw.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError):
            notes.append(f"ignored a vision-model object without a usable bbox_2d: {raw!r}")
            continue
        if not all(math.isfinite(v) and 0.0 <= v <= BOX_SCALE for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            notes.append(f"ignored a vision-model box outside 0-{BOX_SCALE:.0f} or inverted: {raw!r}")
            continue
        centre_px = ((x1 + x2) / 2 / BOX_SCALE * width, (y1 + y2) / 2 / BOX_SCALE * height)
        x_mm, y_mm = location.to_mat_mm([centre_px])[0]
        label = str(raw.get("label", "")).lower()
        detections.append(VlDetection(label, float(x_mm), float(y_mm), min(1.0, max(0.0, confidence))))
    return detections, notes


def nearest(detections: list[VlDetection], obj: DetectedObject, selector: str | None = None) -> VlDetection | None:
    """The closest detection within MATCH_MM of `obj` (whose label matches `selector`, when given)."""
    close = [
        (math.hypot(d.x_mm - obj.x_mm, d.y_mm - obj.y_mm), d)
        for d in detections
        if selector is None or matches(selector, d.label)
    ]
    close = [(dist, d) for dist, d in close if dist <= MATCH_MM]
    return min(close, key=lambda pair: pair[0])[1] if close else None


COLOUR_WORDS = frozenset(COLOUR_HUES) | {"dark"}  # what the marker view can label a blob


def missing_block(
    selector: str, blobs: list[DetectedObject], detections: list[VlDetection] | None = None
) -> SceneError:
    """A SceneError naming what the markers (and the vision model, when asked) saw instead."""
    seen = [o.label for o in blobs]
    message = f"no {selector} block on the mat; the marker view shows {seen or 'no objects'}"
    if detections:
        message += f"; the vision model names {[d.label for d in detections]}"
    return SceneError(message)


def precheck(blobs: list[DetectedObject], task: TaskSpec, vision: bool) -> None:
    """The cheap geometry-only gate before any vision spend: SceneError when the markers can't support the task.

    A name that is a plain colour must be one of the marker view's blobs. Any other name can only come from the vision
    model, so with one available it is checked after the vision call instead.
    """
    for selector in (task.pick, task.place):
        if any(matches(selector, o.label) for o in blobs):
            continue
        if vision and not words(selector) <= COLOUR_WORDS:
            continue
        raise missing_block(selector, blobs)
    if vision and len(blobs) < 2:
        raise SceneError(f"a task needs two objects on the mat; the marker view shows {[o.label for o in blobs]}")


def find_block(
    blobs: list[DetectedObject], selector: str, detections: list[VlDetection] | None, taken: set[int]
) -> int | None:
    """Index of the blob a name picks out: by the marker view's own label first (the largest, when several match),
    else the blob under the vision model's most confident detection that the name matches."""
    free = [i for i, o in enumerate(blobs) if id(o) not in taken]
    by_label = [i for i in free if matches(selector, blobs[i].label)]
    if by_label:
        return max(by_label, key=lambda i: blobs[i].area_mm2)
    named = sorted((d for d in detections or [] if matches(selector, d.label)), key=lambda d: -d.confidence)
    for det in named:
        near = [(math.hypot(det.x_mm - blobs[i].x_mm, det.y_mm - blobs[i].y_mm), i) for i in free]
        near = [pair for pair in near if pair[0] <= MATCH_MM]
        if near:
            return min(near)[1]
    return None


def task_blocks(
    blobs: list[DetectedObject], selectors: tuple[str, str], detections: list[VlDetection] | None = None
) -> tuple[dict[str, DetectedObject], list[DetectedObject]]:
    """The object each name picks out, keyed by name in order, and the blobs with those objects named for the twin.

    A block found through the vision model's words is renamed to the name asked for (`pomade tin` -> `pomade_tin`),
    because the twin and the success check look blocks up by label. SceneError when a name matches nothing.
    """
    blobs, task = list(blobs), {}
    for selector in selectors:
        index = find_block(blobs, selector, detections, {id(o) for o in task.values()})
        if index is None:
            raise missing_block(selector, blobs, [d for d in detections or [] if not RIG_WORDS & words(d.label)])
        if not matches(selector, blobs[index].label):
            blobs[index] = replace(blobs[index], label=slug(selector))
        task[selector] = blobs[index]
    return task, blobs


def split_distractors(
    blobs: list[DetectedObject], task: dict[str, DetectedObject], warnings: list[str]
) -> list[DetectedObject]:
    """Every blob that is not a task block. A repeat of a task block's label is renamed: the expert and StackCheck look
    task blocks up by name, so a second blue or red can't share it."""
    roles = {id(o) for o in task.values()}
    role_labels = {o.label for o in task.values()}
    distractors = []
    for obj in blobs:
        if id(obj) in roles:
            continue
        if obj.label in role_labels:
            warnings.append(f"extra {obj.label} object at ({obj.x_mm:.0f}, {obj.y_mm:.0f}) mm treated as a distractor")
            obj = replace(obj, label=f"other_{obj.label}")
        distractors.append(obj)
    return distractors


def cross_check(
    task: dict[str, DetectedObject],
    distractors: list[DetectedObject],
    detections: list[VlDetection],
    warnings: list[str],
) -> None:
    """Set each task block's confidence from the vision model (in place, keyed by the name the task gave it) and
    flag disagreements in `warnings`. A detection confirms a block when its words include the task's name for it."""
    for selector, block in task.items():
        match = nearest(detections, block, selector)
        if match:
            task[selector] = replace(block, confidence=match.confidence)
        else:
            task[selector] = replace(block, confidence=UNCONFIRMED_CONFIDENCE)
            warnings.append(
                f"vision model did not confirm the {selector} block at ({block.x_mm:.0f}, {block.y_mm:.0f}) mm"
            )
    modelled = [*task.values(), *distractors]
    for det in detections:
        if RIG_WORDS & set(re.split(r"\W+", det.label)):
            continue
        if not any(math.hypot(det.x_mm - o.x_mm, det.y_mm - o.y_mm) <= MATCH_MM for o in modelled):
            warnings.append(
                f"vision model sees {det.label or 'an object'} at ({det.x_mm:.0f}, {det.y_mm:.0f}) mm "
                "where the marker view has no object"
            )


def fuse(
    observation: SceneObservation,
    location: MatLocation,
    detections: list[VlDetection] | None = None,
    notes: list[str] | None = None,
    task: TaskSpec = DEFAULT_TASK,
) -> MarkerScene:
    """The task's two blocks and the distractors in mat millimetres from the marker view, cross-checked by the vision
    model. `objects` is [pick, place, *distractors]; a block named through the vision model's words is renamed to the
    task's name for it.

    `detections=None` means the vision model was not consulted: confidences stay at 1.0, and the task's names must
    match the marker view's own labels (colours). Raises SceneError when a name matches no object. Every other object
    stays in the scene as a distractor, because a policy that never saw clutter in rehearsal can score far worse on a
    cluttered real scene.
    """
    blobs = list(observation.objects)
    roles, blobs = task_blocks(blobs, (task.pick, task.place), detections)
    warnings = list(notes or [])
    distractors = split_distractors(blobs, roles, warnings)
    if detections is not None:
        cross_check(roles, distractors, detections, warnings)
    pick, place = roles[task.pick], roles[task.place]
    markers = SceneObservation(location.markers_found, [pick, place])
    fit = max(0.0, 1.0 - location.residual_mm / MARKER_TOLERANCE_MM)
    return MarkerScene(
        [pick, place, *distractors],
        location.markers_found,
        location.residual_mm,
        Fidelity.perception_from(markers) * fit,
        warnings,
    )
