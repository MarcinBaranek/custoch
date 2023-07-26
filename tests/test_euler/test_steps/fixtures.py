import pytest
from numba import cuda
import numpy


@pytest.fixture
def diffusion_kernel(request):
    wiener_dim: int = request.param

    @cuda.jit(device=True)
    def diffusion_kernel(t, x, out):
        for i in range(x.shape[0]):
            for j in range(wiener_dim):
                out[i, j] = (i + 1) * (j + 1) * x[i, 0] / 10

    setattr(diffusion_kernel, 'wiener_dim', request.param)
    return diffusion_kernel


@cuda.jit(device=True)
def drift_kernel(t, x, out):
    for i in range(x.shape[0]):
        out[i, 0] = (i + 1) * x[i, 0] / 4


def drift_expected(x, dt):
    return numpy.array(
        [[(i + 1) * x[i, 0] / 4 * dt] for i in range(x.shape[0])]
    )
