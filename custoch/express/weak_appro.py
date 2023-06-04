from typing import Callable

from numba import cuda

from ..euler.euler_steps import EulerPath
from ..typing import *
from ..precision import Precisions
from .. import ArgsHandler, KernelManager


class WeakApprox:
    """Class for weak approximation of the SDE."""
    def __init__(
        self,
        drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None],
        diffusion_kernel: Callable[[Time, Vector[d, One], Out[d, m]], None],
        wiener_dim: m,
        dim: d = 2,
        precision: Precisions = Precisions.float64,
        t_0: float = 0.,
        T: float = 1.,
        N: int = 100,
        K: int = 10,
        only_last: bool = True
    ) -> None:
        self.only_last = only_last
        self.n = N
        self.kernel = EulerPath(
            drift_function=drift_function, diffusion_kernel=diffusion_kernel,
            wiener_dim=wiener_dim, dim=dim, precision=precision, t_0=t_0, T=T,
            N=N
        ).get_kernel(only_last=only_last, with_off_set=not only_last)
        self.k = K
        self.dim = dim
        self.precision = precision

    def get_kernel(self):
        return self.get_kernel_only_last() if self.only_last\
            else self.get_kernel_whole_trajectory()

    def get_wrapped_kernel(self, state=None):
        arg_handler = ArgsHandler(
            state=state if state else True, precision=self.precision.value
        )
        arg_handler.add_args(
            0, False, shape=(self.dim, 1), precision=self.precision.value,
            name='start point'
        )
        arg_handler.add_args(
            1, True,
            shape=(
                self.dim if self.only_last else self.dim * self.k,
                self.k if self.only_last else self.n
            ),
            precision=self.precision.value,
            name='result'
        )
        return KernelManager(
            self.get_kernel(), arg_handler, is_device=False, is_random=True,
            n_args=3
        )

    def get_kernel_only_last(self):
        path_kernel = self.kernel
        k = self.k
        dim = self.dim
        precision = self.precision.value

        @cuda.jit
        def __weak_approx(start_point, result, state):
            pos = cuda.grid(1)
            if pos < k:
                tmp_out = cuda.local.array(shape=(dim, 1), dtype=precision)
                path_kernel(start_point, tmp_out, state)
                for i in range(dim):
                    result[i, pos] = tmp_out[i, 0]

        return __weak_approx

    def get_kernel_whole_trajectory(self):
        path_kernel = self.kernel
        k = self.k
        dim = self.dim

        @cuda.jit
        def __weak_approx(start_point, result, state):
            pos = cuda.grid(1)
            if pos < k:
                path_kernel(start_point, pos * dim, result, state)
        return __weak_approx