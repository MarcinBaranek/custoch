from numba import cuda

from .config import WeakApproxConfig
from .euler import EulerPath


class ExperimentKernel(WeakApproxConfig):

    def get_kernel(self):
        euler_path = EulerPath.from_object(self).get_kernel()

        @cuda.jit
        def kernel(result, initial_point, state):
            pos: int = cuda.grid(1)
            if pos < result.size:
                result[pos] = euler_path(initial_point, state)
        return kernel
