from typing import Callable

from numba import cuda

from ....linalg import scale
from ....typing import *
from ..base_step import BaseStep


__all__ = ('EulerDriftStep',)


class EulerDriftStep(BaseStep):
    @staticmethod
    def get_kernel(
            drift_function: Callable[
                [Time, Vector[d, One], Out[d, One]],
                None
            ]
    ) -> Callable[
        [Time, Vector[d, One], float, Out[d, One]], None
    ]:
        @cuda.jit(device=True)
        def __euler_drift_step_out(
                time: Time, point: cuda.device_array,  dt: float,
                out: cuda.device_array
        ):
            drift_function(time, point, out)
            scale(dt, out)

        return __euler_drift_step_out
