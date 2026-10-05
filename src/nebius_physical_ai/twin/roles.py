"""What the pick and place objects are called when the prompt is not always `blue cube` and `red cube`.

An object is described the way the camera sees it: a colour word plus a noun for its shape (`teal can`), or the real
object's own name (`pomade tin`). The same split is used everywhere so no evaluation can silently train on its
held-out objects: the colours and real objects below never appear in a training episode, and the benchmark draws its
held-out suite from exactly them.
"""

import colorsys

from nebius_physical_ai.twin.scene import RGBA

# Colours that read clearly on the grey mat (white would vanish into the paper) and keep their hue apart under the
# randomiser's +-0.03 hue jitter. Blocks are painted relative to the paper (0.6 grey), so a new colour's brightest
# channel stays at or under 0.6: above that the randomiser's brightness jitter clips it and shifts its hue. The held-out
# colours sit in the gaps between the trained hues (mint between green and blue, lime between yellow and green), clear
# of how far the trained colours' own clipped hues drift: orange landed on red, teal on blue, and pink on red too.
# Six trained colours are exactly enough for a scene's two roles and four clutter objects, none sharing a colour.
PALETTE: dict[str, tuple[float, float, float]] = {
    **{name: rgba[:3] for name, rgba in RGBA.items()},
    "black": (0.12, 0.12, 0.12),
    "mint": (0.1, 0.6, 0.4),
    "lime": (0.25, 0.6, 0.1),
}
TRAIN_COLOURS = ("red", "blue", "green", "yellow", "purple", "black")
HELD_OUT_COLOURS = ("mint", "lime")
# Real objects measured with `npai lift`: the tin is seen in training (as a place target and a distractor); the
# glasses case and the ball cluster are only ever seen by the benchmark.
HELD_OUT_REAL = ("glasses_case", "ball_cluster")

BOX_NOUNS = ("box", "block")
SQUARE_BOX_NOUNS = ("cube", *BOX_NOUNS)  # only a near-square footprint is a cube
CYLINDER_NOUNS = ("cylinder", "can")
CUBE_ASPECT = 1.1  # longest over shortest side up to which a box is called a cube


def hue(rgb: tuple[float, float, float]) -> float:
    return colorsys.rgb_to_hsv(*rgb)[0]


def nouns_for(shape: str, length_mm: float, width_mm: float) -> tuple[str, ...]:
    """The words that honestly describe a footprint: cylinders and cans, cubes only when square, else boxes."""
    if shape == "cylinder":
        return CYLINDER_NOUNS
    return SQUARE_BOX_NOUNS if length_mm <= CUBE_ASPECT * width_mm else BOX_NOUNS


def spoken(label: str) -> str:
    """A twin label (`pomade_tin`, `teal_can`) as the words used in a prompt."""
    return label.replace("_", " ")


def instruction(pick_label: str, place_label: str) -> str:
    """The prompt for a pair of twin labels, in the wording of the original `Pick blue cube and place on red cube`."""
    return f"Pick {spoken(pick_label)} and place on {spoken(place_label)}"
