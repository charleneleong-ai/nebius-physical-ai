import pytest

from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec, matches, phrase, slug
from nebius_physical_ai.twin.rollout import STACK_TASK


class TestNames:
    @pytest.mark.parametrize(
        ("selector", "label", "expected"),
        [
            ("blue", "blue", True),
            ("BLUE", "Blue", True),
            ("ball", "multicolored ball cluster", True),  # a word of the model's longer name
            ("pomade tin", "pomade_tin", True),  # underscores separate words
            ("ball cluster", "multicolored ball cluster", True),
            ("cluster ball", "multicolored ball cluster", True),  # order does not matter
            ("blue", "blueberry", False),  # whole words only
            ("tin", "tinted case", False),
            ("pomade tin", "tin", False),  # every word of the selector must be there
            ("yellow", "dark", False),
            ("", "anything", False),
            ("  ", "anything", False),
        ],
    )
    def test_a_selector_matches_when_all_its_words_are_in_the_label(
        self, selector: str, label: str, expected: bool
    ) -> None:
        assert matches(selector, label) is expected

    def test_phrase_and_slug_keep_word_order_and_drop_punctuation(self) -> None:
        assert phrase("Pomade-Tin, (small)") == ["pomade", "tin", "small"]
        assert slug("Pomade Tin") == "pomade_tin" and slug("glasses_case") == "glasses_case"


class TestTaskSpec:
    def test_the_default_is_the_checkpoints_own_task_word_for_word(self) -> None:
        assert (DEFAULT_TASK.pick, DEFAULT_TASK.place) == ("blue", "red")
        assert DEFAULT_TASK.instruction == STACK_TASK and DEFAULT_TASK.in_distribution

    def test_the_standard_names_in_any_case_are_still_the_standard_task(self) -> None:
        task = TaskSpec("Blue", "RED")
        assert task.instruction == STACK_TASK and task.in_distribution

    def test_other_names_derive_an_instruction_from_their_words(self) -> None:
        task = TaskSpec("Pomade  Tin", "glasses_case")
        assert task.instruction == "Pick pomade tin and place on glasses case"
        assert not task.in_distribution

    def test_an_explicit_prompt_wins_and_leaves_the_checkpoints_distribution(self) -> None:
        task = TaskSpec(prompt="stack the cubes")
        assert task.instruction == "stack the cubes" and not task.in_distribution

    def test_the_checkpoint_prompt_on_other_objects_is_still_out_of_distribution(self) -> None:
        task = TaskSpec("pomade tin", "red", prompt=STACK_TASK)
        assert task.instruction == STACK_TASK and not task.in_distribution

    @pytest.mark.parametrize(("pick", "place"), [("red", "red"), ("pomade_tin", "Pomade Tin"), ("Ball", "ball")])
    def test_picking_an_object_onto_itself_is_rejected(self, pick: str, place: str) -> None:
        with pytest.raises(ValueError, match="same object"):
            TaskSpec(pick, place)

    @pytest.mark.parametrize(("pick", "place"), [("", "red"), ("blue", "  "), ("!!", "red")])
    def test_an_empty_name_is_rejected(self, pick: str, place: str) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            TaskSpec(pick, place)

    def test_the_out_of_distribution_note_says_what_was_trained_and_what_is_asked(self) -> None:
        note = TaskSpec("pomade tin", "red").out_of_distribution_note()
        assert STACK_TASK in note and "Pick pomade tin and place on red" in note
