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
    TBD :)
    Currently we refer to examples.
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
