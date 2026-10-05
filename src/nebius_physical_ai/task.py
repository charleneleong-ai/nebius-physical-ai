"""What Preflight is asked to do, and how a name the operator gives picks out an object on the mat.

A task is "put the `pick` object on the `place` object". Each is a name: a colour the marker view labels blobs with
(`blue`, `yellow`, `dark`), or any words the vision model uses for the object (`pomade tin`, `ball`). Matching is by
whole words, so `ball` finds `multicolored ball cluster` and `blue` never finds `blueberry`.
"""

import re
from dataclasses import dataclass

from nebius_physical_ai.twin.rollout import STACK_TASK

DEFAULT_PICK, DEFAULT_PLACE = "blue", "red"
WORD_SPLIT = re.compile(r"[\W_]+")  # underscores separate words: pomade_tin is "pomade tin"


def phrase(text: str) -> list[str]:
    """Lower-case words of a label or selector, in order."""
    return [w for w in WORD_SPLIT.split(text.lower()) if w]


def words(text: str) -> frozenset[str]:
    return frozenset(phrase(text))


def matches(selector: str, label: str) -> bool:
    """Every word of the selector is a word of the label."""
    wanted = words(selector)
    return bool(wanted) and wanted <= words(label)


def slug(text: str) -> str:
    """A selector as an object name the twin can use: `Pomade tin` -> `pomade_tin`."""
    return "_".join(phrase(text))


@dataclass(frozen=True)
class TaskSpec:
    """Put the `pick` object on the `place` object, as the policy is told in `instruction`.

    The default is the one task the GR00T checkpoints were fine-tuned on, with its prompt word for word. Anything else
    is a different scene for the policy, which `in_distribution` reports.
    """

    pick: str = DEFAULT_PICK
    place: str = DEFAULT_PLACE
    prompt: str | None = None  # the instruction sent to the policy; None derives it from the names

    def __post_init__(self) -> None:
        if not words(self.pick) or not words(self.place):
            raise ValueError("a task needs a name for the object to pick and the object to place it on")
        if words(self.pick) == words(self.place):
            raise ValueError(f"pick and place name the same object: {self.pick!r}")

    @property
    def standard_names(self) -> bool:
        return (slug(self.pick), slug(self.place)) == (DEFAULT_PICK, DEFAULT_PLACE)

    @property
    def instruction(self) -> str:
        if self.prompt:
            return self.prompt
        if self.standard_names:
            return STACK_TASK
        return f"Pick {' '.join(phrase(self.pick))} and place on {' '.join(phrase(self.place))}"

    @property
    def in_distribution(self) -> bool:
        """True for the checkpoints' own task: the blue and red cubes and the verbatim prompt."""
        return self.standard_names and self.instruction == STACK_TASK

    def out_of_distribution_note(self) -> str:
        return (
            f"the policy was fine-tuned only on {STACK_TASK!r} with blue and red cubes; rehearsing "
            f"{self.instruction!r} on {self.pick} and {self.place} is outside what it was trained on"
        )


DEFAULT_TASK = TaskSpec()
