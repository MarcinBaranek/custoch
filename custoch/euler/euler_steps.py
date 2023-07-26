from typing import Callable, Any, NoReturn
from dataclasses import dataclass
from ..linalg import add, write_from_to
from ..precision import Precisions
from ..typing import *
from numba import cuda
from .steps.drift import EulerDriftStep
from .steps.diffusion import ClassicalEulerDiffusionStep
# from .steps import EulerStep

__all__ = ('EulerStep', 'EulerDriftStep', 'EulerPath')


def raise_type_error(name: str, variable: Any) -> NoReturn:
    raise TypeError(
        f'`{name}` should be bool, got: {variable} with type: {type(variable)}'
    )


@dataclass(slots=True, frozen=True)
class EulerStep:
    drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None]
    diffusion_kernel: Callable[[Time, Vector[d, One], Out[d, m]], None]
    wiener_dim: m
    dim: d = 2
    precision: Precisions = Precisions.float64

    def get_kernel(
            self, with_user_dw: bool = False
    ) -> Callable[
            [Time, Vector[d, One], float, Out[d, One], RandomState], None
    ] | Callable:
        drift_step_kernel = EulerDriftStep.get_kernel(self.drift_function)
        diffusion_step_kernel = ClassicalEulerDiffusionStep(
            self.diffusion_kernel, self.dim, self.wiener_dim, self.precision
        ).get_kernel()
        dim = self.dim
        precision = self.precision.value
        args = (drift_step_kernel, diffusion_step_kernel, dim, precision)
        match with_user_dw:
            case True: return self.get_euler_step_with_user_dw(*args)
            case False: return self.get_euler_step(*args)
            case _: raise_type_error('with_user_dw', with_user_dw)

    @staticmethod
    def get_euler_step(
            drift_step_kernel: Callable[
                [Time, Vector[d, One], float, Out[d, One]], None
            ],
            diffusion_step_kernel: Callable[
                [Time, Vector[d, One], float, Out[d, One], RandomState], None
            ],
            dim: int,
            precision: str
    ) -> Callable[
            [Time, Vector[d, One], float, Out[d, One], RandomState],
            None
    ]:
        @cuda.jit(device=True)
        def __euler_step(
                time: float, point: cuda.device_array, dt: float,
                out: cuda.device_array, state: cuda.device_array
        ):
            tmp_drift = cuda.local.array(shape=(dim, 1), dtype=precision)
            drift_step_kernel(time, point, dt, tmp_drift)
            diffusion_step_kernel(time, point, dt, out, state)
            add(tmp_drift, out, out)
            add(point, out, out)

        return __euler_step

    @staticmethod
    def get_euler_step_with_user_dw(*args):
        raise NotImplementedError()


@dataclass(slots=True, frozen=True)
class EulerPath(EulerStep):
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
        euler_step = super(self.__class__, self).get_kernel(with_user_dw=False)
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
