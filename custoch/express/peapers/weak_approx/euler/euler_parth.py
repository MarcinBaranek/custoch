import math
from typing import Callable

from numba import cuda

from custoch.linalg import write_from_to, fill, scale, add
from custoch.metrics import sse
from custoch.typing import *
from custoch.random import UniformGenerator, NormalGenerator
from .config import Config
from .euler_step import EulerStep


class EulerPath(Config):

    def get_kernel(self) -> Callable[[Vector[d, One], RandomState], float]:
        uniform_generator = UniformGenerator.get_kernel(self.precision)
        normal_generator = NormalGenerator.get_kernel(self.precision)
        perturbed_wiener_func = self.perturbed_wiener
        dim = self.dim
        wiener_dim = self.wiener_dim
        precision = self.precision
        start_time = self.start_of_time
        grid_size = self.grid_size
        dense_factor = self.dense_factor
        dt_for_exact_alg = self.dt_for_exact
        delta = self.delta
        alpha = 1.,
        beta = 0.5
        euler_step = EulerStep.from_object(self).get_kernel()

        @cuda.jit(device=True)
        def euler_path(
                initial_point: Vector[d, One], state: RandomState
        ) -> float:
            # Declaration
            wiener = cuda.local.array(
                shape=(wiener_dim, 1), dtype='float64'
            )
            wiener_perturbed = cuda.local.array(
                shape=(wiener_dim, 1), dtype=precision
            )
            wiener_perturbed_last = cuda.local.array(
                shape=(wiener_dim, 1), dtype=precision
            )
            dw = cuda.local.array(
                shape=(wiener_dim, 1), dtype='float64'
            )
            dw_perturbed = cuda.local.array(
                shape=(wiener_dim, 1), dtype=precision
            )
            temp_point = cuda.local.array(shape=(dim, 1), dtype='float64')
            temp_point_from_alg = cuda.local.array(
                shape=(dim, 1), dtype=precision
            )

            # Initialization
            write_from_to(initial_point, temp_point)
            write_from_to(initial_point, temp_point_from_alg)

            fill(wiener, 0)
            fill(dw, 0)

            fill(wiener_perturbed_last, 0)
            fill(wiener_perturbed, 0)
            fill(dw_perturbed, 0)

            counter = 0
            cur_time = start_time
            cur_time_for_alg = start_time
            for i in range(grid_size * dense_factor):
                counter = counter + 1
                xi_exact = uniform_generator(
                    cur_time, cur_time + dt_for_exact_alg, state
                )
                normal_generator(dw, state)
                scale(math.sqrt(dt_for_exact_alg), dw)
                add(wiener, dw, wiener)
                euler_step(
                    temp_point, xi_exact, cur_time, dw,
                    dt_for_exact_alg, 0., state
                )
                if counter % dense_factor == 0:
                    counter = 0
                    perturbed_wiener_func(
                        cur_time_for_alg, wiener, wiener_perturbed, delta,
                        alpha, beta, state
                    )
                    scale(-1, wiener_perturbed_last)
                    add(wiener_perturbed, wiener_perturbed_last, dw_perturbed)
                    xi_for_alg = uniform_generator(
                        cur_time_for_alg,
                        cur_time_for_alg
                        + dt_for_exact_alg * dense_factor, state
                    )
                    euler_step(
                        temp_point_from_alg, xi_for_alg, cur_time_for_alg,
                        dw_perturbed, dt_for_exact_alg * dense_factor,
                        delta, state
                    )
                    write_from_to(wiener_perturbed, wiener_perturbed_last)
                    cur_time_for_alg =\
                        cur_time_for_alg + dt_for_exact_alg * dense_factor
                cur_time = cur_time + dt_for_exact_alg
            return sse(temp_point, temp_point_from_alg)

        return euler_path
