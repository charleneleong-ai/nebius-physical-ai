import re

import numpy as np
import pytest
from synthetic_scene import CLEAN, CLUTTERED, Block, SyntheticScene, scene

from nebius_physical_ai.capture.mat import MarkersNotFound, ScenePerceiver
from nebius_physical_ai.perception import (
    MARKER_TOLERANCE_MM,
    UNCONFIRMED_CONFIDENCE,
    CosmosSceneGraph,
    SceneError,
    decode_frame,
    fuse,
    precheck,
    vl_detections,
    vl_jpeg_b64,
)
from nebius_physical_ai.task import TaskSpec
from nebius_physical_ai.twin.scene import A3_MAT

PERCEIVER = ScenePerceiver(A3_MAT)


def graph(*objects: dict) -> CosmosSceneGraph:
    return CosmosSceneGraph(list(objects), [], [])


def perceive(s: SyntheticScene):
    observation, location = PERCEIVER.perceive(s.image)
    assert location is not None, observation.error
    return observation, location


def detections(s: SyntheticScene, location, *objects: dict):
    height, width = s.image.shape[:2]
    found, notes = vl_detections(graph(*objects), location, (width, height))
    return found, notes


def at(objects, label: str):
    return next(o for o in objects if o.label == label)


class TestMarkerLocation:
    def test_pixels_map_back_to_mat_millimetres(self) -> None:
        s = scene()
        location = PERCEIVER.locate(s.image)
        truth = [(130, 150), (220, 150), (10, 200), (340, 20)]
        mapped = location.to_mat_mm([s.pixel(x, y) for x, y in truth])
        np.testing.assert_allclose(mapped, truth, atol=1.5)
        assert location.markers_found == [0, 1, 2, 3] and location.residual_mm < 2.0

    def test_residual_flags_a_marker_that_is_off_its_corner(self) -> None:
        clean = PERCEIVER.locate(scene().image)
        buckled = PERCEIVER.locate(scene(marker_shift_mm={2: (-18.0, 0.0)}).image)
        assert clean.residual_mm < 0.6
        assert buckled.residual_mm > 5 * clean.residual_mm and buckled.residual_mm > 1.5

    def test_too_few_markers_raises_with_the_ids_that_were_seen(self) -> None:
        with pytest.raises(MarkersNotFound, match="need 2 corner markers, saw 1") as err:
            PERCEIVER.locate(scene(markers=(1,)).image)
        assert err.value.found == [1]

    def test_perceive_reports_the_error_and_no_location(self) -> None:
        observation, location = PERCEIVER.perceive(scene(markers=(3,)).image)
        assert location is None and not observation.ok and "saw 1" in observation.error

    def test_observe_is_the_observation_half_of_perceive(self) -> None:
        s = scene()
        assert PERCEIVER.observe(s.image) == PERCEIVER.perceive(s.image)[0]


class TestVisionBoxProjection:
    def test_boxes_land_on_the_true_mat_position(self) -> None:
        s = scene()
        _, location = perceive(s)
        found, notes = detections(s, location, *(s.vl_object(b) for b in CLUTTERED))
        assert notes == []
        for block in CLUTTERED:
            det = at(found, block.label)
            assert (det.x_mm, det.y_mm) == pytest.approx((block.x_mm, block.y_mm), abs=3.0)

    def test_unusable_boxes_are_skipped_with_a_note_each(self) -> None:
        s = scene()
        _, location = perceive(s)
        good = s.vl_object(CLEAN[0])
        found, notes = detections(
            s,
            location,
            good,
            {"label": "blue"},  # no box
            {"label": "red", "bbox_2d": [100, 100, 1280, 200]},  # pixels, not 0-1000
            {"label": "x", "bbox_2d": [500, 500, 400, 600]},  # inverted
            {"label": "y", "bbox_2d": "nope"},
        )
        assert [d.label for d in found] == ["blue"] and len(notes) == 4

    def test_confidence_is_clamped_to_the_unit_interval(self) -> None:
        s = scene()
        _, location = perceive(s)
        found, _ = detections(s, location, s.vl_object(CLEAN[0], confidence=7.0), s.vl_object(CLEAN[1], confidence=-1))
        assert [d.confidence for d in found] == [1.0, 0.0]


class TestFuse:
    def test_marker_geometry_with_no_vision_model(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        result = fuse(observation, location)
        assert [o.label for o in result.objects] == ["blue", "red"]
        for obj, block in zip(result.objects, CLEAN):
            assert (obj.x_mm, obj.y_mm) == pytest.approx((block.x_mm, block.y_mm), abs=4.0)
            assert obj.confidence == 1.0
        assert result.warnings == []
        assert result.perception == pytest.approx(1 - location.residual_mm / MARKER_TOLERANCE_MM)

    def test_distractors_stay_in_the_scene_after_the_task_blocks(self) -> None:
        observation, location = perceive(scene(CLUTTERED))
        objects = fuse(observation, location).objects
        assert [o.label for o in objects[:2]] == ["blue", "red"]
        assert sorted(o.label for o in objects[2:]) == ["dark", "yellow"]
        yellow, dark = at(objects, "yellow"), at(objects, "dark")
        assert (yellow.x_mm, yellow.y_mm) == pytest.approx((90, 60), abs=4.0) and yellow.length_mm > 30
        assert (dark.x_mm, dark.y_mm) == pytest.approx((290, 70), abs=4.0)

    def test_geometry_comes_from_the_markers_not_from_a_misplaced_vision_box(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        found, notes = detections(s, location, s.vl_object(CLEAN[0], 0.95, at=(60, 40)), s.vl_object(CLEAN[1], 0.9))
        result = fuse(observation, location, found, notes)
        blue, red = result.objects
        assert (blue.x_mm, blue.y_mm) == pytest.approx((130, 150), abs=4.0)  # not (60, 40)
        assert blue.confidence == UNCONFIRMED_CONFIDENCE and red.confidence == pytest.approx(0.9)
        assert any("did not confirm the blue block" in w for w in result.warnings)
        assert any("vision model sees blue at (60, 40)" in w for w in result.warnings)

    def test_confirmed_blocks_take_the_vision_models_confidence(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        found, _ = detections(s, location, s.vl_object(CLEAN[0], 0.8), s.vl_object(CLEAN[1], 0.95))
        result = fuse(observation, location, found)
        assert [o.confidence for o in result.objects] == pytest.approx([0.8, 0.95])
        assert result.warnings == []
        assert result.perception == pytest.approx(0.8 * (1 - location.residual_mm / MARKER_TOLERANCE_MM))

    def test_a_vision_label_must_match_to_confirm_a_block(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        swapped = [{**s.vl_object(CLEAN[0]), "label": "red"}, {**s.vl_object(CLEAN[1]), "label": "blue"}]
        found, _ = detections(s, location, *swapped)
        result = fuse(observation, location, found)
        assert [o.confidence for o in result.objects] == [UNCONFIRMED_CONFIDENCE] * 2

    def test_an_object_only_the_vision_model_sees_is_flagged(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        ghost = s.vl_object(Block("mug", 300, 60, 20))  # glass or low contrast: invisible to colour segmentation
        found, _ = detections(s, location, *(s.vl_object(b) for b in CLEAN), ghost)
        warnings = fuse(observation, location, found).warnings
        assert len(warnings) == 1 and re.search(r"vision model sees mug at \(30[01], 60\)", warnings[0])

    @pytest.mark.parametrize("label", ["marker", "ArUco marker", "corner markers", "robot arm", "gripper"])
    def test_the_rig_is_not_an_unmodelled_object(self, label: str) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        corner = {**s.vl_object(Block("x", 5, 5, 10)), "label": label}  # at the printed marker's corner
        found, _ = detections(s, location, *(s.vl_object(b) for b in CLEAN), corner)
        assert fuse(observation, location, found).warnings == []

    def test_a_farm_toy_is_not_mistaken_for_the_arm(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        toy = {**s.vl_object(Block("x", 300, 60, 20)), "label": "farm animal"}
        found, _ = detections(s, location, *(s.vl_object(b) for b in CLEAN), toy)
        assert len(fuse(observation, location, found).warnings) == 1  # whole words only

    def test_a_repeated_task_colour_becomes_a_distractor(self) -> None:
        small_blue = Block("blue", 60, 190, 12)
        observation, location = perceive(scene((*CLEAN, small_blue)))
        result = fuse(observation, location)
        blue = result.objects[0]
        assert (blue.x_mm, blue.y_mm) == pytest.approx((130, 150), abs=4.0)  # the larger one is the task block
        assert [o.label for o in result.objects[2:]] == ["other_blue"]
        assert any("extra blue object" in w for w in result.warnings)

    def test_missing_task_block_names_what_the_markers_saw(self) -> None:
        observation, location = perceive(scene((Block("blue", 130, 150), Block("yellow", 90, 60, 20))))
        with pytest.raises(SceneError, match=r"no red block.*\['blue', 'yellow'\]|no red block.*\['yellow', 'blue'\]"):
            fuse(observation, location)

    def test_an_empty_mat_is_a_scene_error(self) -> None:
        observation, location = perceive(scene(()))
        with pytest.raises(SceneError, match="no blue block.*no objects"):
            fuse(observation, location)

    def test_fewer_markers_and_a_poor_fit_both_lower_perception(self) -> None:
        def perception(s: SyntheticScene) -> float:
            observation, location = perceive(s)
            return fuse(observation, location).perception

        four = perception(scene(CLEAN))
        three = perception(scene(CLEAN, markers=(0, 1, 2)))
        two = perception(scene(CLEAN, markers=(0, 2)))
        buckled = perception(scene(CLEAN, marker_shift_mm={2: (-18.0, 0.0)}))
        assert three == pytest.approx(0.75 * four, rel=0.1) and two == pytest.approx(0.5 * four, rel=0.1)
        assert buckled < 0.7 * four  # a sheet that does not match its markers is not trusted even with all four

    def test_notes_from_unusable_boxes_reach_the_warnings(self) -> None:
        observation, location = perceive(scene(CLEAN))
        assert fuse(observation, location, [], ["ignored a box"]).warnings[0] == "ignored a box"


class TestTaskSelection:
    def test_colour_names_pick_any_two_blobs_and_the_rest_stay_distractors(self) -> None:
        observation, location = perceive(scene(CLUTTERED))
        objects = fuse(observation, location, task=TaskSpec("yellow", "dark")).objects
        assert [o.label for o in objects[:2]] == ["yellow", "dark"]
        assert sorted(o.label for o in objects[2:]) == ["blue", "red"]
        assert (objects[0].x_mm, objects[0].y_mm) == pytest.approx((90, 60), abs=4.0)

    def test_a_name_only_the_vision_model_uses_picks_the_blob_under_its_box(self) -> None:
        s = scene(CLUTTERED)
        observation, location = perceive(s)
        tin = s.vl_object(Block("pomade tin", 90, 60, 20), 0.8)  # the yellow cylinder, as the model names it
        found, _ = detections(s, location, tin, s.vl_object(CLUTTERED[1], 0.95))
        result = fuse(observation, location, found, task=TaskSpec("pomade tin", "red"))
        pick, place = result.objects[:2]
        assert (pick.label, place.label) == ("pomade_tin", "red")  # renamed: the twin looks blocks up by label
        assert (pick.x_mm, pick.y_mm) == pytest.approx((90, 60), abs=4.0)  # geometry still from the markers
        assert (pick.confidence, place.confidence) == pytest.approx((0.8, 0.95))
        assert sorted(o.label for o in result.objects[2:]) == ["blue", "dark"] and result.warnings == []

    def test_one_word_of_the_vision_models_name_is_enough(self) -> None:
        s = scene(CLUTTERED)
        observation, location = perceive(s)
        found, _ = detections(s, location, s.vl_object(Block("pomade tin", 90, 60, 20)), s.vl_object(CLUTTERED[1]))
        assert fuse(observation, location, found, task=TaskSpec("tin", "red")).objects[0].label == "tin"

    def test_a_name_neither_the_markers_nor_the_vision_model_has_lists_both(self) -> None:
        s = scene(CLUTTERED)
        observation, location = perceive(s)
        found, _ = detections(s, location, s.vl_object(Block("pomade tin", 90, 60, 20)), s.vl_object(CLUTTERED[1]))
        with pytest.raises(
            SceneError, match=r"no mug block.*marker view shows.*vision model names \['pomade tin', 'red'\]"
        ):
            fuse(observation, location, found, task=TaskSpec("mug", "red"))

    def test_two_names_cannot_resolve_to_the_same_object(self) -> None:
        s = scene(CLUTTERED)
        observation, location = perceive(s)
        found, _ = detections(s, location, s.vl_object(Block("pomade tin", 90, 60, 20)))
        with pytest.raises(SceneError, match="no tin block"):  # the one yellow blob is already the pomade tin
            fuse(observation, location, found, task=TaskSpec("pomade tin", "tin"))

    def test_a_longer_vision_label_confirms_a_colour_task(self) -> None:
        s = scene(CLEAN)
        observation, location = perceive(s)
        found, _ = detections(
            s, location, {**s.vl_object(CLEAN[0], 0.8), "label": "blue cube"}, s.vl_object(CLEAN[1], 0.95)
        )
        result = fuse(observation, location, found)
        assert [o.confidence for o in result.objects] == pytest.approx([0.8, 0.95]) and result.warnings == []


class TestPrecheck:
    def blobs(self):
        return perceive(scene(CLEAN))[0].objects  # blue and red

    def test_a_colour_the_markers_do_not_see_fails_even_with_a_vision_model(self) -> None:
        with pytest.raises(SceneError, match="no yellow block"):
            precheck(self.blobs(), TaskSpec("yellow", "red"), vision=True)

    def test_a_name_only_the_vision_model_can_supply_waits_for_it(self) -> None:
        task = TaskSpec("pomade tin", "red")
        precheck(self.blobs(), task, vision=True)  # checked after the vision call
        with pytest.raises(SceneError, match="no pomade tin block"):
            precheck(self.blobs(), task, vision=False)

    def test_the_vision_model_cannot_name_objects_that_are_not_there(self) -> None:
        with pytest.raises(SceneError, match="needs two objects"):
            precheck(self.blobs()[:1], TaskSpec("pomade tin", "mug"), vision=True)


class TestSceneGraphJson:
    def test_issues_default_to_empty_and_a_bare_string_becomes_a_list(self) -> None:
        assert CosmosSceneGraph.from_json({"objects": []}).issues == []
        assert CosmosSceneGraph.from_json({"issues": "blue already on red"}).issues == ["blue already on red"]
        assert CosmosSceneGraph.from_json({"issues": ["a", 2]}).issues == ["a", "2"]


class TestFrameHelpers:
    def test_decode_round_trips_a_frame(self) -> None:
        s = scene(CLEAN)
        assert decode_frame(s.b64()).shape == s.image.shape

    @pytest.mark.parametrize(
        ("payload", "message"), [("***", "not valid base64"), ("aGVsbG8=", "not a decodable image")]
    )
    def test_decode_rejects_non_images(self, payload: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            decode_frame(payload)

    def test_vision_jpeg_is_downscaled_but_keeps_the_aspect_ratio(self) -> None:
        big = np.random.default_rng(42).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)
        height, width = decode_frame(vl_jpeg_b64(big)).shape[:2]
        assert max(height, width) == 1280 and height / width == pytest.approx(2 / 3, abs=0.01)

    def test_small_frames_are_not_resized(self) -> None:
        small = np.zeros((50, 100, 3), np.uint8)
        assert decode_frame(vl_jpeg_b64(small)).shape == small.shape
