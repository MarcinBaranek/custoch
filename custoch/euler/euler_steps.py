import math
from typing import Callable, Any, NoReturn
from dataclasses import dataclass
from ..random import NormalGenerator
from ..linalg import scale, matmul
from ..precision import Precisions
from ..typing import *
from numba import cuda


def raise_type_error(name: str, variable: Any) -> NoReturn:
    raise TypeError(
        f'`{name}` should be bool, got: {variable} with type: {type(variable)}'
    )


class EulerDriftStep:
    @staticmethod
    def get_kernel(
            kernel: Callable[
                [float, Vector[d, One], Out[d, One]],
                None
            ]
    ) -> Callable[[Vector[d, One], float, float, Out[d, One]], None]:
        @cuda.jit(device=True)
        def __euler_drift_step_out(
                point: cuda.device_array, time: float, dt: float,
                out: cuda.device_array
        ):
            kernel(time, point, out)
            scale(dt, out)

        return __euler_drift_step_out


@dataclass(slots=True, frozen=True)
class EulerDiffusionStep:
    function: Callable[
        [float, Vector[d, One], Out[d, m]], None
    ]
    wiener_dim: int
    dim: int = 2
    precision: Precisions = Precisions.float64

    def get_kernel(
            self, with_user_dw: bool = False
    ) -> Callable[
            [Vector[d, One], float, dW, Out[d, One]],
            None
    ] | Callable[
            [Vector[d, One], float, float, Out[d, One], RandomState],
            None
    ]:
        match with_user_dw:
            case True: return self.get_drift_step_user_dw()
            case False: return self.get_drift_step()
            case _: raise raise_type_error('with_user_dw', with_user_dw)

    def get_drift_step_user_dw(self) -> Callable[
            [Vector[d, One], float, dW, Out[d, One]],
            None
    ]:
        kernel = self.function
        dim = self.dim
        wiener_dim = self.wiener_dim
        precision = self.precision.value

        @cuda.jit(device=True)
        def __euler_diffusion_step_with_user_dw(
                point: cuda.device_array, time: float, dw: cuda.device_array,
                out: cuda.device_array
        ):
            tmp = cuda.local.array(shape=(dim, wiener_dim), dtype=precision)
            kernel(time, point, tmp)
            matmul(tmp, dw, out)

        return __euler_diffusion_step_with_user_dw

    def get_drift_step(self) -> Callable[
            [Vector[d, One], float, float, Out[d, One], RandomState],
            None
    ]:
        # TODO check if we get the same kernel for different precision kernel
        #  is not overwritten
        kernel = self.function
        generator = NormalGenerator.get_kernel(self.precision)
        precision = self.precision.value
        dim = self.dim
        wiener_dim = self.wiener_dim

        @cuda.jit(device=True)
        def __euler_diffusion_step(
                point: cuda.device_array, time: float, dt: float,
                out: cuda.device_array, state: cuda.device_array
        ):
            tmp = cuda.local.array(shape=(dim, wiener_dim), dtype=precision)
            dw = cuda.local.array(shape=(wiener_dim, 1), dtype=precision)
            generator(dw, state)
            scale(math.sqrt(dt), dw)
            kernel(time, point, tmp)
            matmul(tmp, dw, out)

        return __euler_diffusion_step


@dataclass(slots=True, frozen=True)
class EulerStep:
    drift_function: Callable[[float, Vector[d, One], Out[d, One]], None]
    diffusion_kernel: Callable[[float, Vector[d, One], Out[d, m]], None]
    wiener_dim: int
    dim: int = 2
    precision: Precisions = Precisions.float64

    def get_kernel(self, with_user_dw: bool = False) -> Callable:
        drift_kernel = EulerDriftStep.get_kernel(self.drift_function)
        diffusion_kernel = EulerDiffusionStep(
            self.drift_function, self.wiener_dim, self.dim, self.precision
        ).get_kernel(with_user_dw)
        dim = self.dim
        precision = self.precision.value
        args = (drift_kernel, diffusion_kernel, dim, precision)
        match with_user_dw:
            case True: return self.get_euler_step(*args)
            case False: return self.get_euler_step(*args)
            case _: raise_type_error('with_user_dw', with_user_dw)

    @staticmethod
    def get_euler_step(
            drift_kernel: Callable[[float, Vector[d, One], Out[d, One]], None],
            diffusion_kernel: Callable[
                [float, Vector[d, One], Out[d, m]], None
            ],
            dim: int,
            precision: str
    ) -> Callable:
        @cuda.jit(device=True)
        def __euler_step(
                point: cuda.device_array, time: float, dt: float,
                out: cuda.device_array, state: cuda.device_array
        ):
            tmp = cuda.local.array(shape=(dim, 1), dtype=precision)


            drift_kernel(point, time, dt, tmp)
            diffusion_kernel(point, point, )

        return __euler_step
