from __future__ import annotations

from typing import Callable, TypeVar, ClassVar
from dataclasses import dataclass

import numpy as np

from custoch.typing import *
from custoch import Precisions

Delta = TypeVar('Delta', bound=float)


@dataclass
class Config:
    drift_function: Callable[
        [Time, Vector[d, One], Out[d, One], Delta, RandomState],
        None
    ] = None
    diffusion_kernel: Callable[
        [Time, Vector[d, One], Out[d, m], Delta, RandomState],
        None
    ] = None
    perturbed_wiener: Callable[
        [Time, Vector[d, m], Out[d, m], Delta, float, float, RandomState],
        None
    ] = None
    initial_point: Vector[d, One] = np.ones(shape=(2, 1))
    wiener_dim: m = 1
    dim: d = 2
    precision: Precisions = Precisions.float64
    end_of_time: T = 1.
    start_of_time: t_0 = 0.
    grid_size: N = 100
    dense_factor: int = 10
    delta: Delta = 0.

    config_fields: ClassVar[tuple[str, ...]] = (
        'drift_function',
        'diffusion_kernel',
        'perturbed_wiener',
        'initial_point',
        'wiener_dim',
        'dim',
        'precision',
        'end_of_time',
        'start_of_time',
        'grid_size',
        'dense_factor',
        'delta'
    )

    @classmethod
    def from_object(cls, obj) -> Config:
        for key in cls.config_fields:
            if key not in dir(obj):
                raise AttributeError(
                    f"To create object of class: {cls.__name__} "
                    f"from object: {obj} is impossible! "
                    f"Object has not attribute: {key}"
                )
        return cls(**{key: getattr(obj, key) for key in cls.config_fields})

    @property
    def dt_for_exact(self) -> float:
        return (self.end_of_time - self.start_of_time) \
               / (self.grid_size * self.dense_factor)
