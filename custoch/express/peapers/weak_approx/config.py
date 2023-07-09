from dataclasses import dataclass
from typing import ClassVar

from .euler.config import Config


@dataclass
class WeakApproxConfig(Config):
    """Perturbed """
    number_of_trajectories: int = 10

    config_fields: ClassVar[tuple[str, ...]] =\
        Config.config_fields + ('number_of_trajectories',)
