import numpy as np
import numpy.testing as npt
import pytest
from numba import cuda

from custoch import KernelManager, State
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.steps.diffusion import ClassicalEulerDiffusionStep
from ....utils import precision, tolerance
from ..fixtures import diffusion_kernel

_ = (precision, diffusion_kernel)


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


@pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('dim', [1, 2, 3, 6])
@pytest.mark.parametrize('diffusion_kernel', [1, 2, 3, 4], indirect=True)
def test_diffusion_step(
        precision: Precisions,
        diffusion_kernel,
        dim: int, dt: float
):
    wiener_dim: int = diffusion_kernel.wiener_dim
    # dw for state with seed 7
    dw = np.array([[0.61972869], [-1.33773295], [-0.44804679], [-0.94795856]])\
         * np.sqrt(dt)
    a = np.ones(shape=(dim, 1))
    diffusion_result = np.zeros(shape=(dim, wiener_dim))
    result = np.zeros(shape=(dim, 1))

    arg_handler = ArgsHandler(state=State(n=1, seed=7), precision=precision)
    arg_handler.add_args(
        1, out=False, shape=(dim, 1), precision=precision
    )
    arg_handler.add_args(3, out=True, shape=(dim, 1), precision=precision)
    kernel = KernelManager(
        ClassicalEulerDiffusionStep(
            diffusion_kernel, dim, wiener_dim, precision
        ).get_kernel(),
        arg_handler,
        is_device=True,
        n_args=5
    )
    kernel[1, 1](0., a, dt, result)
    diffusion_kernel.py_func(0., a, diffusion_result)
    exp_result = diffusion_result @ dw[:wiener_dim, ...]

    npt.assert_allclose(
        result, exp_result, atol=tolerance[precision]
    )
