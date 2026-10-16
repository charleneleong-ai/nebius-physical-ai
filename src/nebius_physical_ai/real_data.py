"""Real SO-101 teleoperation reshaped like the twin's demos, for co-training (scripts/real_mix.py has why)."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

ARMNET = "armnet/armnetbench_v01_lerobot_so101"
FRAME_HW = (480, 640)  # the twin's cameras


def armnet_episodes(episodes: Mapping[str, Sequence[object]], rollouts: bool = False, teleop: bool = True) -> list[int]:
    """The 400 teleoperated reference demos when `teleop`, and the policies' rollouts labelled successful when
    `rollouts`. `episodes` holds ArmnetBench's episode table columns; a missing policy type means teleoperation."""
    keep = []
    for index, policy, outcome in zip(episodes["episode_index"], episodes["policy_type"], episodes["success_class"]):
        is_teleop = not isinstance(policy, str) or policy == "teleoperated"
        if (teleop and is_teleop) or (rollouts and not is_teleop and outcome == "successful"):
            keep.append(int(index))
    return keep


def to_twin_units(joints: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """ArmnetBench records LeRobot's range mode, each joint's calibrated travel as [-100, 100] (gripper [0, 100]), and the
    twin maps the menagerie joint ranges onto the same scale, so the units carry over. `offsets` (same units, from
    `real_mix.py calibrate`) correct the farm arms' hand calibration first; overshoot past the calibrated ends is then
    clipped (shoulder lift reaches -103; wrist roll, a full-turn joint, 155)."""
    out = np.asarray(joints, dtype=np.float64) + (0.0 if offsets is None else offsets)
    out[..., :5] = np.clip(out[..., :5], -100, 100)
    out[..., 5] = np.clip(out[..., 5], 0, 100)
    return out


def resample_positions(n: int, src_fps: int, dst_fps: int) -> np.ndarray:
    """Fractional source-frame position of every output frame spanning the same duration."""
    return np.arange(int((n - 1) * dst_fps / src_fps) + 1) * src_fps / dst_fps


def interpolate(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Rows of `values` (n, d) linearly interpolated at fractional `positions`."""
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo + 1, len(values) - 1)
    w = (positions - lo)[:, None]
    return values[lo] * (1 - w) + values[hi] * w


def fit_frame(image: np.ndarray, hw: tuple[int, int] = FRAME_HW) -> np.ndarray:
    """Centre-crop an (H, W, 3) frame to the twin's aspect, then resize to it."""
    (h, w), (th, tw) = image.shape[:2], hw
    if w * th > h * tw:
        cw = round(h * tw / th)
        image = image[:, (w - cw) // 2 : (w - cw) // 2 + cw]
    else:
        ch = round(w * th / tw)
        image = image[(h - ch) // 2 : (h - ch) // 2 + ch]
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)


def closing_frames(gripper: np.ndarray) -> np.ndarray:
    """Frames where the gripper falls through the middle of its range in the episode: the start of each grasp."""
    open_ = gripper > (gripper.min() + gripper.max()) / 2
    return np.flatnonzero(open_[:-1] & ~open_[1:]) + 1
