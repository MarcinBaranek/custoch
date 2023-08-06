from dataclasses import dataclass
from typing import Callable

from ..typing import *


@dataclass(slots=True, frozen=True)
class Dimensions:
    wiener_dim: m
    space_dim: d = 2


@dataclass(slots=True, frozen=True)
class Coefficients:
    drift: Callable[[Time, Vector[d, One], Out[d, One]], None]
    diffusion: Callable[[Time, Vector[d, One], Out[d, m]], None]


@dataclass(slots=True, frozen=True)
class SDE:
    dims: Dimensions
    coefficients: Coefficients
