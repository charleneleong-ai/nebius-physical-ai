"""NEW_EMBODIMENT modality config for the SO-101 with the wrist roll in a block of its own.

This is the layout the community SO-101 checkpoints (quangnd58/GR00T-N1.6-3B-so101-multitask and the N1.7 twin) were
trained with: `single_arm` is the first four joints, predicted relative to the current pose; `wrist_roll` is predicted
as an absolute target, like the gripper. NVIDIA's examples/SO100/so100_config.py instead keeps the roll inside the
relative `single_arm` block. Fine-tuning a community checkpoint in its own layout keeps the wrist-roll head it learned.

Pass this file to GR00T's launch_finetune.py as --modality-config-path, with a dataset whose meta/modality.json splits
the six joints the same way (`npai demos --layout split`, or `finetune.py link-dataset --layout split`). It only runs
inside a GR00T environment, which provides the `gr00t` package.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ARM_BLOCKS = ["single_arm", "wrist_roll", "gripper"]  # pan..wrist flex, roll, gripper: 4 + 1 + 1 values
HORIZON = 16  # action steps predicted per query


def joint_action(rep: ActionRepresentation) -> ActionConfig:
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)  # joint space, not end-effector


so101_split_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=ARM_BLOCKS),
    "action": ModalityConfig(
        delta_indices=list(range(HORIZON)),
        modality_keys=ARM_BLOCKS,
        action_configs=[
            joint_action(ActionRepresentation.RELATIVE),  # single_arm: deltas from the current pose
            joint_action(
                ActionRepresentation.ABSOLUTE
            ),  # wrist_roll: the follower's reading does not track the command
            joint_action(ActionRepresentation.ABSOLUTE),  # gripper
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(so101_split_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
