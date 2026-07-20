"""Schedulers (diffusion & flow processes) + a tiny name registry for loading."""

from __future__ import annotations

import json
from pathlib import Path

from ..configuration import CONFIG_NAME
from .base import Scheduler, SchedulerConfig, make_betas
from .ddim import DDIMConfig, DDIMScheduler
from .ddpm import DDPMConfig, DDPMScheduler
from .euler import EulerConfig, EulerDiscreteScheduler
from .flow_match_euler import FlowMatchConfig, FlowMatchEulerScheduler
from .trellis_flow import TrellisFlowEulerSampler, TrellisFlowSample

#: Maps the config class name stored in config.json -> scheduler class.
SCHEDULERS: dict[str, type[Scheduler]] = {
    "DDPMConfig": DDPMScheduler,
    "DDIMConfig": DDIMScheduler,
    "EulerConfig": EulerDiscreteScheduler,
    "FlowMatchConfig": FlowMatchEulerScheduler,
}


def load_scheduler(path: str | Path) -> Scheduler:
    """Load a scheduler from a directory containing ``config.json``.

    The concrete class is selected from the config's ``_class_name`` tag written
    at save time, falling back to DDPM when absent.
    """
    path = Path(path)
    config_path = path / CONFIG_NAME if path.is_dir() else path
    data = json.loads(config_path.read_text())
    cls = SCHEDULERS.get(data.get("_class_name", "DDPMConfig"))
    if cls is None:
        raise ValueError(f"Unknown scheduler config {data.get('_class_name')!r}.")
    return cls(cls.config_class.from_dict(data))


__all__ = [
    "Scheduler",
    "SchedulerConfig",
    "make_betas",
    "DDPMScheduler",
    "DDPMConfig",
    "DDIMScheduler",
    "DDIMConfig",
    "EulerDiscreteScheduler",
    "EulerConfig",
    "FlowMatchEulerScheduler",
    "FlowMatchConfig",
    "TrellisFlowEulerSampler",
    "TrellisFlowSample",
    "SCHEDULERS",
    "load_scheduler",
]
