from typing import Optional

import numpy as np
from numba import cuda
from numpy.typing import NDArray, ArrayLike

from custoch.precision import BasePrecision

__all__ = ('ArrayHandler',)


class ArrayHandler(BasePrecision):
    """Class for handling array during call kernel.

    Provide interface to copy content of the given array between host and
    device.

    ...

    Attributes
    ----------
    array : Optional[NDArray]
        Content of the input array.

    Methods
    -------
    to_device()
        Send array to device returning cuda device array.
    to_host()
        Copy content of the device array to `array` and return them.

    Examples
    --------
    >>> handler = ArrayHandler(array=[1, 2, 3])
    >>> handler.array
    array([1., 2., 3.])
    >>> handler.to_host()
    Traceback (most recent call last):
    ...
    RuntimeError: No data was sent to the device!
    >>> handler.to_device() # doctest: +ELLIPSIS
    <numba.cuda.cudadrv.devicearray.DeviceNDArray object at ...>
    >>> handler.to_host()
    array([1., 2., 3.])
    """
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
        if not isinstance(value, tuple | list):
            raise TypeError(
                f'Shape should be instance of tuple got: {value} '
                f'with type: {type(value)}'
            )
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
            self._array = None
            return
        self._array = np.array(value, dtype=self.precision)
        if self.shape:
            assert self._array.shape == self._shape,\
                f'Array shape should be {self.shape}, got {self._array.shape}'

    def to_device(self) -> cuda.device_array:
        """Send current content of the `array` to device and return device
        array object."""
        if self._array is None:
            raise RuntimeError(
                'Array is still None and could not be sent to the device!'
            )
        self.device_array = cuda.to_device(self._array)
        return self.device_array

    def to_host(self) -> NDArray:
        """Copy current content of the `device_array` to host returning result.
        """
        if self.device_array is None:
            raise RuntimeError('No data was sent to the device!')
        self.device_array.copy_to_host(self._array)
        return self._array
