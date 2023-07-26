from abc import ABCMeta
from typing import Callable
from dataclasses import dataclass
from ....precision import Precisions
from ....typing import *
from ..base_step import BaseStep


@dataclass(slots=True, frozen=True)
class BaseEulerDiffusionStep(BaseStep, metaclass=ABCMeta):
    function: Callable[
        [Time, Vector[d, One], Out[d, m]], None
    ]
    dim: d = 2
    wiener_dim: m = 2
    precision: Precisions = Precisions.float64
