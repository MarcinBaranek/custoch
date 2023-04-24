from typing import Optional

import numpy as np
from numba import cuda
from numpy.typing import NDArray, ArrayLike

from custoch.precision import BasePrecision


class ArrayHandler(BasePrecision):
    array: Optional[NDArray]
    shape: Optional[tuple[int, int]]
    device_array: Optional[cuda.device_array]

    def __init__(
            self,
            array: Optional[ArrayLike] = None,
            shape: Optional[tuple[int, int]] = None,
            precision: str = 'float64'
    ):
        super().__init__(precision=precision)
        self.shape = shape
        self.array = array
        self.device_array = None

    @property
    def shape(self) -> Optional[tuple[int, int]]:
        return self._shape

    @shape.setter
    def shape(self, value: Optional[tuple[int, int]]) -> None:
        if value is None:
            self._shape = None
            return
        assert len(value) <= 2,\
            f'Maximal allowed shape\'s dimension is 2, got shape: {value}'
        assert len(value) >= 1,\
            f'Minimal allowed shape\'s dimension is 1, got shape: {value}'
        for item in value:
            assert item > 0, 'Shape should have positive coefficients.'
        self._shape = tuple(map(int, value))

    @property
    def array(self) -> NDArray:
        return self._array

    @array.setter
    def array(self, value: ArrayLike):
        if value is None:
            self._array = value
            return
        self._array = np.array(value, dtype=self.precision)
        if self.shape:
            assert self._array.shape == self._shape,\
                f'Array shape should be {self.shape}, got {self._array.shape}'

    def to_device(self) -> cuda.device_array:
        if self._array is None:
            raise RuntimeError(
                f'Array is still None and could be sent to device!'
            )
        self.device_array = cuda.to_device(self._array)
        return self.device_array

    def to_host(self) -> NDArray:
        if self.device_array is None:
            raise RuntimeError(f'No data was sent to the device!')
        self.device_array.copy_to_host(self._array)
        return self._array
