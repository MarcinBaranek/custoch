from numba import cuda

from ..abstraction import WeakApproxAbstraction
from ....euler.euler_steps import EulerPath
from .base import AbstractWeakApproxStrategy


class WeakApproxWholeTrajectoriesStrategy(AbstractWeakApproxStrategy):
    def get_internal_kernel(self, weak_approx_config: WeakApproxAbstraction):
        return EulerPath(
            drift_function=weak_approx_config.drift_function,
            diffusion_kernel=weak_approx_config.diffusion_kernel,
            wiener_dim=weak_approx_config.wiener_dim,
            dim=weak_approx_config.dim,
            precision=weak_approx_config.precision,
            t_0=weak_approx_config.start_of_time,
            T=weak_approx_config.end_of_time,
            N=weak_approx_config.grid_size
        ).get_kernel(only_last=False, with_off_set=True)

    def get_kernel(self, weak_approx_config: WeakApproxAbstraction):
        path_kernel = self.get_internal_kernel(weak_approx_config)
        number_of_trajectories = weak_approx_config.number_of_trajectories
        dim = weak_approx_config.dim

        @cuda.jit
        def __weak_approx(start_point, result, state):
            pos: int = cuda.grid(1)
            if pos < number_of_trajectories:
                path_kernel(start_point, pos * dim, result, state)

        return __weak_approx

    def get_state_shape(self, weak_approx_config: WeakApproxAbstraction):
        first_dim = \
            weak_approx_config.dim * weak_approx_config.number_of_trajectories
        return first_dim, weak_approx_config.grid_size
