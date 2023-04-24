from typing import Optional, Callable

from .args_handler import ArgsHandler
from .wrapped_device_function import get_device_function_wrapper


class KernelManager:
    def __init__(
            self,
            kernel,
            args_handler: Optional[ArgsHandler],
            is_device: bool = False,
            n_args: int = 1
    ) -> None:
        self.kernel = get_device_function_wrapper(kernel, n_args) \
            if is_device else kernel
        self.args_handler = args_handler if args_handler else ArgsHandler()

    def __getitem__(self, grid) -> Callable[[...], None]:
        def caller(*args):
            device_args = self.args_handler(*args, grid=grid)
            self.kernel[grid](*device_args)
            self.args_handler.copy_to_host()

        return caller
