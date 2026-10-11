import runpy
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from nebius_physical_ai.policy.gr00t import SPLIT

CONFIG = Path(__file__).parent.parent / "scripts" / "modality_configs" / "so101_split.py"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


@dataclass
class ModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]
    action_configs: list[Any] | None = None


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: Any
    format: Any


@dataclass
class Registry:
    registered: list[tuple[dict[str, ModalityConfig], Any]] = field(default_factory=list)


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModalityConfig]:
    """Runs the config file against a stand-in for the `gr00t` package and returns what it registered."""
    registry = Registry()

    def module(name: str, **attrs: Any) -> ModuleType:
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    kinds = {
        name: type(name, (), {"DEFAULT": "default", "NON_EEF": "non_eef"}) for name in ("ActionFormat", "ActionType")
    }
    module("gr00t")
    module("gr00t.configs")
    module("gr00t.configs.data")
    module(
        "gr00t.configs.data.embodiment_configs",
        register_modality_config=lambda config, embodiment_tag: registry.registered.append((config, embodiment_tag)),
    )
    module("gr00t.data")
    module("gr00t.data.embodiment_tags", EmbodimentTag=type("EmbodimentTag", (), {"NEW_EMBODIMENT": "new_embodiment"}))
    module(
        "gr00t.data.types",
        ActionConfig=ActionConfig,
        ActionRepresentation=ActionRepresentation,
        ModalityConfig=ModalityConfig,
        **kinds,
    )
    runpy.run_path(str(CONFIG))
    assert len(registry.registered) == 1
    config, tag = registry.registered[0]
    assert tag == "new_embodiment"
    return config


def test_the_config_registers_the_split_layout_the_client_and_datasets_use(
    registered: dict[str, ModalityConfig],
) -> None:
    assert registered["state"].modality_keys == list(SPLIT.keys)
    assert registered["action"].modality_keys == list(SPLIT.keys)
    assert registered["video"].modality_keys == ["front", "wrist"]
    assert registered["language"].modality_keys == ["annotation.human.task_description"]


def test_only_the_arm_block_is_relative_and_the_roll_and_gripper_are_absolute(
    registered: dict[str, ModalityConfig],
) -> None:
    action = registered["action"]
    reps = dict(zip(action.modality_keys, (c.rep for c in action.action_configs)))
    assert reps == {
        "single_arm": ActionRepresentation.RELATIVE,
        "wrist_roll": ActionRepresentation.ABSOLUTE,
        "gripper": ActionRepresentation.ABSOLUTE,
    }
    assert len(action.action_configs) == len(action.modality_keys)  # one config per block, in order


def test_it_predicts_sixteen_steps_from_the_current_frame_only(registered: dict[str, ModalityConfig]) -> None:
    assert registered["action"].delta_indices == list(range(16))
    assert registered["state"].delta_indices == [0] and registered["video"].delta_indices == [0]
