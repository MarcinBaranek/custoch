from numba import cuda

from ..abstraction import WeakApproxAbstraction
from ....euler.euler_steps import EulerPath
from .base import AbstractWeakApproxStrategy


class WeakApproxOnlyLastStrategy(AbstractWeakApproxStrategy):
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
        ).get_kernel(only_last=True, with_off_set=False)

    def get_kernel(self, weak_approx_config: WeakApproxAbstraction):
        path_kernel = self.get_internal_kernel(weak_approx_config)
        number_of_trajectories = weak_approx_config.number_of_trajectories
        dim = weak_approx_config.dim
        precision = weak_approx_config.precision.value

        @cuda.jit
        def __weak_approx(start_point, result, state):
            pos = cuda.grid(1)
            if pos < number_of_trajectories:
                tmp_out = cuda.local.array(shape=(dim, 1), dtype=precision)
                path_kernel(start_point, tmp_out, state)
                for i in range(dim):
                    result[i, pos] = tmp_out[i, 0]

        return __weak_approx

    def get_state_shape(self, weak_approx_config: WeakApproxAbstraction):
        return \
            weak_approx_config.dim, weak_approx_config.number_of_trajectories
