from typing import Optional, Callable

from .args_handler import ArgsHandler
from .wrapped_device_function import get_device_function_wrapper


class KernelManager:
    """Class for wrapping CUDA kernels.

    This wrapper simplifies the interfaces of kernels.
    The wrapper is handling under the hood
        * Coping arrays between host and device
        * Random state if kernel needs it.

    Examples
    --------
    Firstly we need to have a cuda kernel
    >>> from numba import cuda
    >>> import numpy as np
    >>> from custoch.typing import Vector
    >>> @cuda.jit
    ... def arr_sum(left: Vector, right: Vector, out: Vector):
    ...     idx = cuda.grid(1)
    ...     if idx < left.shape[0]:
    ...         out[idx] = left[idx] + right[idx]

    Now to call the kernel, we need to keep in the mind the memroy stuff.
    For this we can use the `KernelManager`
    >>> wrapped_kernel = KernelManager(arr_sum, n_args=3)
    >>> x, y, z = np.ones(3), 3 * np.ones(3), np.zeros(3)
    >>> wrapped_kernel[1, 3](x, y, z)
    >>> z
    array([4., 4., 4.])

    The whole logic with tranfer data between host and device is done under
    the hood by `KernelManager`.

    To have much more higer control we can use `ArgsHandler` also.

    >>> from custoch.kernel_manager import ArgsHandler
    >>> arg_handler = ArgsHandler(state=False, to_out=(3,))
    >>> wrapped_kernel = KernelManager(
    ...     arr_sum, args_handler=arg_handler, n_args=3
    ... )
    >>> z *= 0
    >>> wrapped_kernel[1, 3](x, y, z)
    >>> z
    array([4., 4., 4.])

    To add validatio data use `ArrayHandler`

    >>> from custoch.kernel_manager import ArrayHandler
    >>> arg_handler = ArgsHandler(state=False)

    Add metadata about arguments. Index indicate positional position in kernel,
    out if array should be copied from device to host. Shape indicate expected
    shape of the input array. Name is a string, only for debug purporse.

    >>> arg_handler.add_args(index=0, out=False, shape=(3,), name='x')
    >>> arg_handler.add_args(index=1, out=False, shape=(3,), name='x')
    >>> arg_handler.add_args(index=2, out=True, shape=(3,), name='z')

    The result is as before

    >>> wrapped_kernel[1, 3](x, y, z)
    >>> z
    array([4., 4., 4.])

    But if we provide array with another shape that (3,)

    >>> x = np.ones(5)
    >>> wrapped_kernel[1, 3](x, y, z)
    Traceback (most recent call last):
        ...
    AssertionError: Array shape should be (3,), got (5,)
    Array name: x

    To the error message is added the debug name of array.
    """
    def __init__(
            self,
            kernel,
            args_handler: Optional[ArgsHandler] = None,
            is_device: bool = False,
            is_random: bool = False,
            n_args: int = 1
    ) -> None:
        self.kernel = get_device_function_wrapper(kernel, n_args) \
            if is_device else kernel
        self.args_handler = args_handler if args_handler else is_random

    def __getitem__(self, grid) -> Callable:
        def caller(*args):
            if isinstance(self.args_handler, bool):
                self.args_handler = ArgsHandler.create_from_args(
                    *args, state=self.args_handler
                )
            device_args = self.args_handler(*args, grid=grid)
            self.kernel[grid](*device_args)
            self.args_handler.copy_to_host()

        return caller
