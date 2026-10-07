import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.cli import app
from nebius_physical_ai.task import TaskSpec, matches
from nebius_physical_ai.twin import roles
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler
from nebius_physical_ai.twin.randomize import (
    DISTRACTOR_LABELS,
    Episode,
    Randomization,
    SceneRandomizer,
    outline,
    radius,
)
from nebius_physical_ai.twin.rollout import STACK_TASK, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout

LAYOUT = TwinLayout()
REAL = [
    DetectedObject(**o)
    for o in json.loads((Path(__file__).parent.parent / "data/capture/real_objects.json").read_text())
]
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


def draw(n: int, seed: int = 42, **kwargs) -> list[Episode]:
    kwargs = {"real_objects": REAL, "role_agnostic": True} | kwargs
    randomizer, sampler = SceneRandomizer(LAYOUT, seed=seed, **kwargs), SceneSampler(LAYOUT, seed=seed)
    return [randomizer.episode(sampler) for _ in range(n)]


def described(n: int, seed: int = 42, **kwargs) -> list[Episode]:
    """Only role-agnostic episodes: the original share is switched off."""
    return draw(n, seed, ranges=Randomization(original_share=0.0, **kwargs.pop("ranges", {})), **kwargs)


def colour_word(label: str) -> str:
    return label.split("_")[0]


class TestPrompts:
    def test_the_prompt_names_each_role_object_and_no_other(self) -> None:
        for e in described(200):
            names = [o.label for o in e.objects]
            assert e.prompt == TaskSpec(roles.spoken(e.pick), roles.spoken(e.place)).instruction
            assert names[:2] == [e.pick, e.place] and len(set(names)) == len(names)
            for role in (e.pick, e.place):  # the words of the prompt select exactly one object in the scene
                assert [n for n in names if matches(roles.spoken(role), n)] == [role]

    def test_colour_words_are_never_shared_inside_a_scene(self) -> None:
        for e in described(200):
            words = [colour_word(o.label) for o in e.objects if colour_word(o.label) in roles.PALETTE]
            assert len(words) == len(set(words))

    def test_about_a_quarter_keep_the_original_cubes_and_prompt(self) -> None:
        episodes = draw(600)
        original = [e for e in episodes if e.prompt == STACK_TASK]
        assert 0.19 < len(original) / len(episodes) < 0.31
        assert all((e.pick, e.place) == ("blue", "red") for e in original)
        assert all({"blue", "red"} <= {o.label for o in e.objects} for e in original)
        assert all(e.prompt != STACK_TASK for e in episodes if e not in original)

    def test_a_randomizer_that_is_not_role_agnostic_always_teaches_the_one_task(self) -> None:
        episodes = draw(60, role_agnostic=False)
        assert {(e.pick, e.place, e.prompt) for e in episodes} == {("blue", "red", STACK_TASK)}

    def test_the_task_argument_is_the_prompt_of_original_episodes(self) -> None:
        randomizer = SceneRandomizer(LAYOUT, seed=42)
        assert randomizer.episode(SceneSampler(LAYOUT, seed=42), "Stack it").prompt == "Stack it"

    def test_the_same_seed_gives_the_same_episodes_and_another_seed_does_not(self) -> None:
        assert draw(40, seed=42) == draw(40, seed=42)
        assert [e.prompt for e in draw(40, seed=42)] != [e.prompt for e in draw(40, seed=43)]


class TestDescriptions:
    def test_nouns_fit_the_shapes_they_describe(self) -> None:
        seen = set()
        for e in described(400):
            for o in e.objects:
                colour, _, noun = o.label.partition("_")
                if colour not in roles.PALETTE or noun in ("tin", "case", "cluster"):
                    continue
                seen.add(noun)
                if noun in roles.CYLINDER_NOUNS:
                    assert o.shape == "cylinder"
                else:
                    assert o.shape == "box"
                if noun == "cube":
                    assert o.length_mm <= roles.CUBE_ASPECT * o.width_mm
        assert seen == {*roles.CYLINDER_NOUNS, "cube", *roles.BOX_NOUNS}

    def test_each_object_looks_like_the_colour_it_is_called(self) -> None:
        chromatic = {
            c: roles.hue(roles.PALETTE[c]) for c in (*roles.TRAIN_COLOURS, *roles.HELD_OUT_COLOURS) if c != "black"
        }

        def nearest(rgb: tuple[float, float, float]) -> str:
            h = roles.hue(rgb)
            return min(chromatic, key=lambda c: min(abs(h - chromatic[c]), 1 - abs(h - chromatic[c])))

        checked = 0
        for kwargs in ({}, {"colours": roles.HELD_OUT_COLOURS, "exclude_real": ()}):
            for e in described(150, **kwargs):
                for o in e.objects[:2]:
                    colour = colour_word(o.label)
                    if colour not in roles.PALETTE:
                        continue  # a real object, painted as measured
                    checked += 1
                    if colour == "black":
                        assert max(o.rgb) < 0.4
                    else:
                        assert nearest(o.rgb) == colour
        assert checked > 200


class TestHeldOut:
    def training_names(self, n: int = 400) -> set[str]:
        return {o.label for e in draw(n) for o in e.objects}

    def test_training_episodes_never_contain_the_held_out_colours_or_real_objects(self) -> None:
        names = self.training_names()
        assert not {n for n in names if colour_word(n) in roles.HELD_OUT_COLOURS}
        assert not names & set(roles.HELD_OUT_REAL)
        assert "pomade_tin" in names  # the tin is the one real object training does see

    def test_the_benchmark_side_does_draw_them(self) -> None:
        """Without the exclusions the same code does produce them, so the test above is not vacuous."""
        names = {
            o.label
            for e in described(400, colours=roles.HELD_OUT_COLOURS, exclude_real=(), ranges={"real_place_share": 0.5})
            for o in e.objects
        }
        assert {n for n in names if colour_word(n) in roles.HELD_OUT_COLOURS}
        assert set(roles.HELD_OUT_REAL) <= names

    def test_a_held_out_colour_is_never_used_as_clutter(self) -> None:
        for e in described(300, colours=roles.HELD_OUT_COLOURS, exclude_real=()):
            assert {colour_word(o.label) for o in e.objects[2:]} - set(roles.PALETTE) <= {"pomade", "glasses", "ball"}
            assert not {colour_word(o.label) for o in e.objects[2:]} & set(roles.HELD_OUT_COLOURS)

    def test_the_two_held_out_sets_are_disjoint_from_training_and_each_other(self) -> None:
        assert not set(roles.TRAIN_COLOURS) & set(roles.HELD_OUT_COLOURS)
        assert set(roles.TRAIN_COLOURS) | set(roles.HELD_OUT_COLOURS) <= set(roles.PALETTE)


class TestRealObjects:
    def test_real_objects_are_place_targets_only_and_about_a_third_of_the_time(self) -> None:
        episodes = described(900)
        assert not {e.pick for e in episodes} & {o.label for o in REAL}  # nothing real fits the jaws
        on_tin = [e for e in episodes if e.place == "pomade_tin"]
        assert 0.25 < len(on_tin) / len(episodes) < 0.42
        tin = next(o for o in REAL if o.label == "pomade_tin")
        for e in on_tin:
            placed = e.objects[1]
            assert 0.9 * tin.width_mm - 0.1 <= placed.width_mm <= 1.1 * tin.width_mm + 0.1  # measured size +-10%
            assert placed.shape == tin.shape and placed.rgb == tin.rgb

    def test_place_on_forces_that_object(self) -> None:
        glasses = next(o for o in REAL if o.label == "glasses_case")
        randomizer = SceneRandomizer(LAYOUT, seed=42, real_objects=REAL, role_agnostic=True, exclude_real=())
        sampler = SceneSampler(LAYOUT, seed=42)
        for _ in range(30):
            e = randomizer.described(sampler, place_on=glasses)
            assert e.place == "glasses_case" and e.prompt.endswith("place on glasses case")
            assert e.pick != "glasses_case"


class TestClutter:
    def test_clutter_spans_zero_to_four_objects_up_to_the_real_sizes_and_never_overlaps(self) -> None:
        episodes = described(500)
        counts = {len(e.objects) - 2 for e in episodes}
        assert counts == {0, 1, 2, 3, 4}
        clutter = [o for e in episodes for o in e.objects[2:]]
        assert max(o.length_mm for o in clutter) > 100 and max(o.height_mm for o in clutter) > 55
        assert (
            max(o.length_mm for o in clutter) <= 140 * 1.1 + 0.1 and max(o.height_mm for o in clutter) <= 65 * 1.1 + 0.1
        )
        for e in episodes[:150]:
            for o in e.objects[2:]:
                edge = outline(o, np.array([o.x_mm, o.y_mm]))
                for task_object in e.objects[:2]:
                    gap = np.hypot(*(edge - [task_object.x_mm, task_object.y_mm]).T).min()
                    assert gap > radius(task_object) + Randomization().clearance_mm

    def test_original_episodes_get_the_larger_clutter_too(self) -> None:
        episodes = [e for e in draw(500) if e.prompt == STACK_TASK]
        sizes = [o.length_mm for e in episodes for o in e.objects[2:]]
        assert max(sizes) > 100
        assert {o.label for e in episodes for o in e.objects[2:] if o.label in DISTRACTOR_LABELS}  # legacy names

    def test_without_role_agnostic_the_clutter_keeps_its_old_limits(self) -> None:
        episodes = draw(300, role_agnostic=False)
        assert max(len(e.objects) - 2 for e in episodes) <= 2
        blocks = [o for e in episodes for o in e.objects[2:] if o.label in DISTRACTOR_LABELS]  # not the real copies
        assert blocks and max(o.height_mm for o in blocks) <= 45


class TestLookalikes:
    """Pick-sized decoys of the pick's shape, told apart only by the prompt's colour word."""

    @pytest.mark.parametrize("original_share", [0.0, 1.0])
    def test_nearly_every_scene_has_a_decoy_shaped_like_the_pick_in_another_colour(self, original_share: float) -> None:
        ranges = Randomization(original_share=original_share, lookalikes=(1, 2))
        episodes = draw(200, ranges=ranges)
        widest = ranges.pick_width_mm[1] + 0.1
        decoyed = 0
        for e in episodes:
            pick = e.objects[0]
            decoyed += any(
                o.shape == pick.shape and o.width_mm <= widest and colour_word(o.label) != colour_word(pick.label)
                for o in e.objects[2:]
            )
            if original_share == 0.0:  # described prompts must still single out one object per colour word
                words = [colour_word(o.label) for o in e.objects if colour_word(o.label) in roles.PALETTE]
                assert len(words) == len(set(words))
        assert decoyed >= 0.95 * len(episodes)

    @pytest.mark.parametrize(("role_agnostic", "fingerprint"), [(True, "ddd27a8752f0"), (False, "d980e9d05a24")])
    def test_without_lookalikes_a_seed_draws_the_scenes_it_always_did(
        self, role_agnostic: bool, fingerprint: str
    ) -> None:
        scenes = [
            (o.label, round(float(o.x_mm), 1), round(float(o.y_mm), 1), round(o.width_mm, 1))
            for e in draw(20, role_agnostic=role_agnostic)
            for o in e.objects
        ]
        assert hashlib.sha1(repr(scenes).encode()).hexdigest()[:12] == fingerprint  # pinned before lookalikes existed


class TestOriginalShare:
    def sampler_and_randomizer(self, share: float | None):
        from nebius_physical_ai.cli import build_randomizer

        sampler = SceneSampler(LAYOUT, seed=42)
        return sampler, build_randomizer(sampler, 42, None, None, "0.3,0.6,0.1", True, share)

    @pytest.mark.parametrize(("share", "low", "high"), [(0.5, 0.42, 0.58), (0.1, 0.04, 0.16), (None, 0.19, 0.31)])
    def test_the_share_of_original_prompt_episodes_follows_the_option(
        self, share: float | None, low: float, high: float
    ) -> None:
        sampler, randomizer = self.sampler_and_randomizer(share)
        episodes = [randomizer.episode(sampler) for _ in range(500)]
        assert low < sum(e.prompt == STACK_TASK for e in episodes) / len(episodes) < high  # None keeps the default 0.25

    def test_all_original_and_no_original_are_the_two_extremes(self) -> None:
        for share, expected in ((1.0, True), (0.0, False)):
            sampler, randomizer = self.sampler_and_randomizer(share)
            assert {randomizer.episode(sampler).prompt == STACK_TASK for _ in range(60)} == {expected}

    @pytest.mark.parametrize("share", ["1.5", "-0.1"])
    def test_a_share_outside_zero_to_one_is_a_usage_error(self, tmp_path: Path, share: str) -> None:
        args = ["demos", str(tmp_path / "d"), "--randomize", "--role-agnostic", "--original-share", share]
        assert CliRunner().invoke(app, args).exit_code == 2


@pytest.mark.parametrize("flags", [["--role-agnostic"], ["--lookalikes", "2"]])
def test_scene_options_without_randomize_are_a_usage_error(tmp_path: Path, flags: list[str]) -> None:
    result = CliRunner().invoke(app, ["demos", str(tmp_path / "d"), *flags])
    assert result.exit_code == 2 and "--randomize" in " ".join(result.output.split())


@needs_model
class TestWithTheSimulator:
    def test_the_expert_stacks_the_named_pick_on_the_named_place_and_not_the_reverse(self) -> None:
        sampler = SceneSampler(LAYOUT, seed=42)
        randomizer = SceneRandomizer(
            LAYOUT, seed=42, real_objects=REAL, role_agnostic=True, ranges=Randomization(original_share=0.0)
        )
        wins = reversed_wins = 0
        for _ in range(8):
            e = randomizer.episode(sampler)
            with Twin.from_objects(e.objects, LAYOUT, start_pose=sampler.start_pose()) as twin:
                list(PickPlaceExpert(twin, e.prompt, e.pick, e.place, image_size=(24, 32)).run())
                twin.settle(0.5)
                wins += StackCheck(e.pick, e.place)(twin)
                reversed_wins += StackCheck(e.place, e.pick)(twin)
        assert wins >= 5 and reversed_wins == 0

    def test_recorded_episodes_carry_their_own_prompts(self, tmp_path: Path) -> None:
        out = tmp_path / "demos"
        result = CliRunner().invoke(
            app,
            ["demos", str(out), "--episodes", "4", "--seed", "42", "--randomize", "--role-agnostic"]
            + ["--real-objects", str(Path(__file__).parent.parent / "data/capture/real_objects.json")],
        )
        assert result.exit_code == 0, result.output
        tasks = [json.loads(line)["task"] for line in (out / "meta/tasks.jsonl").read_text().splitlines()]
        episodes = [json.loads(line) for line in (out / "meta/episodes.jsonl").read_text().splitlines()]
        assert len(episodes) == 4 and len(set(tasks)) >= 2
        assert all(e["tasks"][0] in tasks for e in episodes)
        assert all(t == STACK_TASK or t.startswith("Pick ") and " and place on " in t for t in tasks)
