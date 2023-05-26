import numpy
import numpy as np
import numpy.testing as npt
from numba import cuda
import pytest

from custoch import KernelManager, State
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.euler_steps import (
    EulerDriftStep, EulerDiffusionStep, EulerStep, EulerPath
)
from ..utils import precision, tolerance

__all__ = ('precision', 'tolerance')


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


@pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('shape', [(3, 1), (2, 1), (6, 1), (1, 1)])
def test_drift_step(
        precision: Precisions, shape: tuple[int], dt: float
):
    a = np.ones(shape=shape)
    result = np.zeros(shape=shape)
    arg_handler = ArgsHandler(state=False, precision=precision)
    arg_handler.add_args(1, out=False, shape=shape, precision=precision)
    arg_handler.add_args(3, out=True, shape=shape, precision=precision)
    kernel = KernelManager(
        EulerDriftStep.get_kernel(drift_kernel),
        arg_handler,
        is_device=True,
        n_args=4
    )
    kernel[1, 1](0., a, dt, result)
    npt.assert_allclose(
        result, drift_expected(a, dt), atol=tolerance[precision]
    )


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
        EulerDiffusionStep(
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


@pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('dim', [1, 3, 2, 6])
@pytest.mark.parametrize('diffusion_kernel', [1, 2, 3, 4], indirect=True)
def test_euler_step(
        precision: Precisions, diffusion_kernel, dim: int, dt: float
):
    wiener_dim: int = diffusion_kernel.wiener_dim
    # dw for state with seed 7
    dw = np.array([[0.61972869], [-1.33773295], [-0.44804679], [-0.94795856]])\
         * np.sqrt(dt)
    a = np.ones(shape=(dim, 1))
    result = np.ones(shape=(dim, 1)) * 10000
    diffusion_result = np.zeros(shape=(dim, wiener_dim))

    arg_handler = ArgsHandler(state=State(n=1, seed=7), precision=precision)
    arg_handler.add_args(
        1, out=False, shape=a.shape, precision=precision, name='Init point'
    )
    arg_handler.add_args(
        3, out=True, shape=result.shape, precision=precision, name='result'
    )
    kernel = KernelManager(
        EulerStep(
            drift_function=drift_kernel, diffusion_kernel=diffusion_kernel,
            wiener_dim=wiener_dim, dim=dim, precision=precision
        ).get_kernel(with_user_dw=False),
        arg_handler,
        is_device=True,
        n_args=5
    )
    kernel[1, 1](0., a, dt, result)
    diffusion_kernel.py_func(0., a, diffusion_result)
    exp_result = diffusion_result @ dw[:wiener_dim, ...]
    exp_result = a + drift_expected(a, dt) + exp_result
    npt.assert_allclose(
        result, exp_result, atol=tolerance[precision]
    )


# @pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
# @pytest.mark.parametrize('shape', [(3, 2), (2, 3), (6, 4)])
@pytest.mark.parametrize('diffusion_kernel', [1, 2, 3, 4], indirect=True)
def test_euler_path(
        precision: Precisions, diffusion_kernel, shape: tuple[int]=(3, 2)
):
    shape = (shape[0], diffusion_kernel.wiener_dim)
    N = 1000
    t_0 = 0.
    T = 1.
    a = np.ones(shape=(shape[0], 1))
    result = np.zeros(shape=(shape[0], N))

    arg_handler = ArgsHandler(state=State(n=1, seed=17), precision=precision)
    arg_handler.add_args(
        0, out=False, shape=a.shape, precision=precision, name='Init point'
    )
    arg_handler.add_args(
        1, out=True, shape=result.shape, precision=precision, name='result'
    )
    kernel = KernelManager(
        EulerPath(
            drift_function=drift_kernel, diffusion_kernel=diffusion_kernel,
            wiener_dim=shape[1], dim=shape[0], precision=precision, t_0=t_0,
            T=T, N=N
        ).get_kernel(with_user_dw=False),
        arg_handler,
        is_device=True,
        n_args=3
    )
    kernel[1, 1](a, result)
    # import matplotlib.pyplot as plt
    # plt.plot(result.T)
    # plt.show()
