import math
from typing import Callable
from dataclasses import dataclass

from numba import cuda

from .....random import NormalGenerator
from .....linalg import scale, matmul
from .....typing import *
from ..abstraction import BaseEulerDiffusionStep


@dataclass(slots=True, frozen=True)
class ClassicalEulerDiffusionStep(BaseEulerDiffusionStep):
    def get_kernel(self) -> Callable[
        [Time, Vector[d, One], float, Out[d, One], RandomState], None
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
                time: Time, point: cuda.device_array, dt: float,
                out: cuda.device_array, state: cuda.device_array
        ):
            tmp = cuda.local.array(shape=(dim, wiener_dim), dtype=precision)
            dw = cuda.local.array(shape=(wiener_dim, 1), dtype=precision)
            generator(dw, state)
            scale(math.sqrt(dt), dw)
            kernel(time, point, tmp)
            matmul(tmp, dw, out)

        return __euler_diffusion_step
