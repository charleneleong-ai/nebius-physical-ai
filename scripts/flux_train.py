"""lerobot-train for FLUX 3 Action with its conditioning or action representation changed.

The SO-101 checkpoint conditions on its own 8-step state and command history and predicts command deltas. Switching
to current-frame conditioning or absolute joint targets changes the width of the action-conditioning heads, which
flux3's strict restore refuses: `strict=False` is its switch for re-initialising embodiment heads only (a trunk
mismatch still raises), the way BFL gave its DROID model fresh absolute-joint heads. The checkpoint's saved
processors are its history/delta pipeline, so any other configuration builds its own from the dataset statistics,
with lerobot-train's rename map and device applied. History-conditioned runs go through unchanged.

    ~/lerobot-flux/.venv/bin/python scripts/flux_train.py --config_path=lora.json \\
        --policy.path=black-forest-labs/flux-3-action-so101 --policy.conditioning=frame \\
        --policy.condition_on_past_actions=false --policy.action_representation=absolute \\
        --policy.delta_absolute_dims=[] --policy.n_obs_steps=1 --policy.history_snapshots=1 ...
"""

import sys
from typing import Any

from lerobot.policies.flux3 import modeling_flux3, processor_flux3
from lerobot.scripts.lerobot_train import main

checkpoint_loader = modeling_flux3.Flux3Policy.from_pretrained.__func__
saved_processors = processor_flux3.make_flux3_pre_post_processors_from_pretrained


def from_pretrained(cls: type, pretrained_name_or_path: str, **kwargs: Any) -> modeling_flux3.Flux3Policy:
    kwargs.setdefault("strict", False)
    return checkpoint_loader(cls, pretrained_name_or_path, **kwargs)


def processors_from_pretrained(config: Any, pretrained_path: str, **kwargs: Any) -> tuple[Any, Any]:
    if config.conditioning == "history":
        return saved_processors(config, pretrained_path, **kwargs)
    pre, post = processor_flux3.make_flux3_pre_post_processors(config, kwargs.get("dataset_stats"))
    overrides = kwargs.get("preprocessor_overrides") or {}
    settings = {**overrides.get("rename_observations_processor", {}), **overrides.get("device_processor", {})}
    for step in pre.steps:
        for attr, value in settings.items():
            if hasattr(step, attr):
                setattr(step, attr, value)
    return pre, post


if __name__ == "__main__":  # the guard matters: lerobot's dataloader workers spawn and re-import this file
    modeling_flux3.Flux3Policy.from_pretrained = classmethod(from_pretrained)
    processor_flux3.make_flux3_pre_post_processors_from_pretrained = processors_from_pretrained
    sys.exit(main())
