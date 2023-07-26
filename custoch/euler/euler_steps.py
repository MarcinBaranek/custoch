from typing import Callable, Any, NoReturn
from dataclasses import dataclass
from ..linalg import write_from_to
from ..precision import Precisions
from ..typing import *
from numba import cuda
from .steps.diffusion import ClassicalEulerDiffusionStep
from .steps import EulerStep

__all__ = ('EulerPath',)


def raise_type_error(name: str, variable: Any) -> NoReturn:
    raise TypeError(
        f'`{name}` should be bool, got: {variable} with type: {type(variable)}'
    )


@dataclass(slots=True, frozen=True)
class EulerStepCLS:
    drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None]
    diffusion_kernel: Callable[[Time, Vector[d, One], Out[d, m]], None]
    wiener_dim: m
    dim: d = 2
    precision: Precisions = Precisions.float64

    def get_kernel(
            self
    ) -> Callable[
            [Time, Vector[d, One], float, Out[d, One], RandomState], None
    ] | Callable:
        return EulerStep(
            drift_function=self.drift_function,
            diffusion_step=ClassicalEulerDiffusionStep(
                self.diffusion_kernel, self.dim,
                self.wiener_dim, self.precision
            )
        ).get_kernel()


@dataclass(slots=True, frozen=True)
class EulerPath(EulerStepCLS):
    t_0: float = 0.
    T: float = 1.
    N: int = 100

    def get_kernel(
            self,
            with_user_dw: bool = False,
            only_last: bool = False,
            with_off_set: bool = False
    ) -> Callable[
            [Time, Vector[d, One], float, Out[d, One], RandomState], None
    ] | Callable:
        euler_step = super(self.__class__, self).get_kernel()
        dt = (self.T - self.t_0) / self.N
        if only_last:
            return self.get_euler_path_only_last(
                euler_step, self.dim, self.precision.value, self.N, self.t_0,
                dt
            )
        elif with_off_set:
            return self.get_euler_path_with_off_set(
                euler_step, self.dim, self.precision.value, self.N, self.t_0,
                dt
            )
        else:
            return self.get_euler_path(
                euler_step, self.dim, self.precision.value, self.N, self.t_0,
                dt
            )

    @staticmethod
    def get_euler_path(
            euler_step: Callable[
                [Time, Vector[d, One], float, Out[d, One], RandomState],
                None
            ], dim, precision, n_steps: int, start_time: float, dt: float
    ):
        @cuda.jit(device=True)
        def __euler_path(
                point: cuda.device_array, out: cuda.device_array,
                state: cuda.device_array
        ):
            tmp_in = cuda.local.array(shape=(dim, 1), dtype=precision)
            tmp_out = cuda.local.array(shape=(dim, 1), dtype=precision)
            write_from_to(point, tmp_in)
            cur_time = start_time
            for i in range(n_steps):
                euler_step(cur_time, tmp_in, dt, tmp_out, state)

                cur_time = cur_time + dt
                # COLLECTOR
                for j in range(dim):
                    out[j, i] = tmp_out[j, 0]
                write_from_to(tmp_out, tmp_in)
        return __euler_path

    @staticmethod
    def get_euler_path_with_off_set(
            euler_step: Callable[
                [Time, Vector[d, One], float, Out[d, One], RandomState],
                None
            ], dim, precision, n_steps: int, start_time: float, dt: float
    ):
        @cuda.jit(device=True)
        def __euler_path_with_off_set(
                point: cuda.device_array, off_set, out: cuda.device_array,
                state: cuda.device_array
        ):
            tmp_in = cuda.local.array(shape=(dim, 1), dtype=precision)
            tmp_out = cuda.local.array(shape=(dim, 1), dtype=precision)
            write_from_to(point, tmp_in)
            cur_time = start_time
            for i in range(n_steps):
                euler_step(cur_time, tmp_in, dt, tmp_out, state)

                cur_time = cur_time + dt
                # COLLECTOR
                for j in range(dim):
                    out[j + off_set, i] = tmp_out[j, 0]
                write_from_to(tmp_out, tmp_in)

        return __euler_path_with_off_set

    @staticmethod
    def get_euler_path_only_last(
            euler_step: Callable[
                [Time, Vector[d, One], float, Out[d, One], RandomState],
                None
            ], dim, precision, n_steps: int, start_time: float, dt: float
    ):
        @cuda.jit(device=True)
        def __euler_path_only_last(
                point: cuda.device_array, out: cuda.device_array,
                state: cuda.device_array
        ):
            tmp_in = cuda.local.array(shape=(dim, 1), dtype=precision)
            tmp_out = cuda.local.array(shape=(dim, 1), dtype=precision)
            write_from_to(point, tmp_in)
            cur_time = start_time
            for i in range(n_steps):
                euler_step(cur_time, tmp_in, dt, tmp_out, state)
                cur_time = cur_time + dt
            # COLLECTOR
            write_from_to(tmp_out, out)

        return __euler_path_only_last
