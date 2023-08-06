from typing import Callable
from dataclasses import dataclass

from numba import cuda

from .....linalg import matmul
from .....typing import *
from ..abstraction import BaseEulerDiffusionStep


@dataclass(slots=True, frozen=True)
class EulerDiffusionStepWithUserDW(BaseEulerDiffusionStep):

    def get_kernel(self) -> Callable[
        [Time, Vector[d, One], dW, Out[d, One]], None
    ]:
        kernel = self.function
        dim: d = self.dim
        wiener_dim: m = self.wiener_dim
        precision = self.precision.value

        @cuda.jit(device=True)
        def __euler_diffusion_step_with_user_dw(
                time: Time, point: cuda.device_array, dw: cuda.device_array,
                out: cuda.device_array
        ):
            tmp = cuda.local.array(shape=(dim, wiener_dim), dtype=precision)
            kernel(time, point, tmp)
            matmul(tmp, dw, out)

        return __euler_diffusion_step_with_user_dw
