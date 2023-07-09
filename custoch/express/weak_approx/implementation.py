from ... import ArgsHandler, KernelManager
from .abstraction import WeakApproxAbstraction
from .strategies import AbstractWeakApproxStrategy


class WeakApprox(WeakApproxAbstraction):
    """Class for weak approximation of the SDE."""

    def get_wrapped_kernel(
            self, strategy: AbstractWeakApproxStrategy, state=None
    ):
        arg_handler = ArgsHandler(
            state=state if state else True, precision=self.precision.value
        )
        arg_handler.add_args(
            0, False, shape=(self.dim, 1), precision=self.precision.value,
            name='start point'
        )
        arg_handler.add_args(
            1, True,
            shape=strategy.get_state_shape(self),
            precision=self.precision.value,
            name='result'
        )
        kernel = strategy.get_kernel(self)
        return KernelManager(
            kernel, arg_handler, is_device=False, is_random=True,
            n_args=3
        )
