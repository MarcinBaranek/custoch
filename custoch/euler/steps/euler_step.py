from dataclasses import dataclass
from typing import Callable

from numba import cuda

from ...typing import *
from ...linalg import add
from .drift import EulerDriftStep
from .diffusion import BaseEulerDiffusionStep
from .base_step import BaseStep


@dataclass(slots=True, frozen=True)
class EulerStep(BaseStep):
    drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None]
    diffusion_step: BaseEulerDiffusionStep

    def get_kernel(
            self, with_user_dw: bool = False
    ) -> Callable[
            [Time, Vector[d, One], float, Out[d, One], RandomState], None
    ] | Callable:
        drift_step_kernel = EulerDriftStep.get_kernel(self.drift_function)
        diffusion_step_kernel = self.diffusion_step.get_kernel()
        dim = self.diffusion_step.dim
        precision = self.diffusion_step.precision.value

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
