from enum import Enum

import numpy as np
from numba.core.types import float16, float32, float64, Float

__all__ = ('Precisions', 'BasePrecision')


class Precisions(str, Enum):
    float16: str = 'float16'
    float32: str = 'float32'
    float64: str = 'float64'

    def __contains__(self, item: str) -> bool:
        return item in tuple(Precisions)

    def __str__(self):
        return self.value


class BasePrecision:
    """Class for Handling precision"""
    precision: str

    def __init__(self, precision: str = 'float64', *args, **kwargs):
        if precision == np.float32:
            precision = 'float32'
        if precision == np.float64:
            precision = 'float64'
        self.precision = Precisions(precision)

    @property
    def precision(self) -> type:
        return getattr(np, self._precision)

    @precision.setter
    def precision(self, value: str) -> None:
        if value not in tuple(Precisions):
            raise ValueError(
                f'Precision should be one of {tuple(Precisions)}, got: {value}'
            )
        self._precision = value

    def __str__(self):
        return self._precision

    @staticmethod
    def numba_type(precision: str) -> Float:
        match precision:
            case 'float16': return float16
            case 'float32': return float32
            case 'float64': return float64
            case _:
                raise ValueError(
                    f'Given precision: {precision} is not supported. '
                    f'Use one of the {tuple(map(str, Precisions))}.'
                )
