from typing import Callable

from ...typing import *
from ...precision import Precisions


class StrongApproxConfig:
    drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None] = None
    diffusion_kernel: Callable[[Time, Vector[d, One], Out[d, m]], None] = None
    wiener_dim: m = 1
    dim: d = 2
    precision: Precisions = Precisions.float64
    end_of_time: T = 1.
    start_of_time: t_0 = 0.
    grid_size: N = 100
    number_of_trajectories: int = 10

    fields = tuple(
        key for key in locals()
        if not key.startswith('_') or key == 'fields'
    )
