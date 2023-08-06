from numba import cuda

from custoch.linalg import scale, add, matmul
from .config import Config, Delta
from custoch.typing import *


class EulerStep(Config):
    def get_kernel(self):
        drift = self.drift_function
        diffusion = self.diffusion_kernel
        dim = self.dim
        wiener_dim = self.wiener_dim
        precision = self.precision.value

        @cuda.jit(device=True)
        def euler_step(
                point: Vector[d, One], xi: float, cur_time: Time,
                dw: dW, dt: float, disruptive_factor: Delta, state: RandomState
        ):
            tmp = cuda.local.array(shape=(dim, 1), dtype=precision)
            tmp_drift_res = cuda.local.array(shape=(dim, 1),
                                             dtype=precision)
            tmp_drift = cuda.local.array(
                shape=(dim, wiener_dim), dtype=precision
            )
            drift(xi, point, tmp, disruptive_factor, state)
            scale(dt, tmp)

            diffusion(cur_time, point, tmp_drift, disruptive_factor, state)
            matmul(tmp_drift, dw, tmp_drift_res)
            add(point, tmp, point)
            add(point, tmp_drift_res, point)
        return euler_step
