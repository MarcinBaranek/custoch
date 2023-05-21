import numpy
import numpy as np
import numpy.testing as npt
from numba import cuda
import pytest

from custoch import KernelManager, State
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.euler_steps import (
    EulerDriftStep, EulerDiffusionStep, EulerStep
)
from ..utils import precision, tolerance

__all__ = ('precision', 'tolerance')


@cuda.jit(device=True)
def drift_kernel(t, x, out):
    for i in range(x.shape[0]):
        out[i, 0] = (i + 1) * x[i, 0] / 4


def drift_expected(x, dt):
    return numpy.array(
        [[(i + 1) * x[i, 0] / 4 * dt] for i in range(x.shape[0])]
    )


@pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('shape', [(3, 1), (2, 1), (6, 1)])
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


@cuda.jit(device=True)
def diffusion_kernel(t, x, out):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = (i + 1) * (j + 1) * x[i, 0] / 4


@pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('shape', [(3, 2), (2, 3), (6, 4)])
def test_diffusion_step(
        precision: Precisions, shape: tuple[int], dt: float
):
    # dw for state with seed 7
    dw = np.array([[0.61972869], [-1.33773295], [-0.44804679], [-0.94795856]])\
         * np.sqrt(dt)
    a = np.ones(shape=(shape[0], 1))
    exp_result = np.zeros(shape=shape)
    result = np.zeros(shape=(shape[0], 1))

    arg_handler = ArgsHandler(state=State(n=1, seed=7), precision=precision)
    arg_handler.add_args(
        1, out=False, shape=(shape[0], 1), precision=precision
    )
    arg_handler.add_args(3, out=True, shape=(shape[0], 1), precision=precision)
    kernel = KernelManager(
        EulerDiffusionStep(
            diffusion_kernel, shape[0], shape[1], precision
        ).get_kernel(),
        arg_handler,
        is_device=True,
        n_args=5
    )
    kernel[1, 1](0., a, dt, result)

    diffusion_kernel.py_func(0., a, exp_result)
    exp_result = exp_result @ dw[:shape[1], ...]

    npt.assert_allclose(
        result, exp_result, atol=tolerance[precision]
    )


# @pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
# @pytest.mark.parametrize('shape', [(3, 2), (2, 3), (6, 4)])
def test_euler_step(
        precision: Precisions, shape: tuple[int]=(3, 2), dt: float = 1.e-4
):
    a = np.ones(shape=(shape[0], 1))
    result = np.zeros(shape=(shape[0], 1))

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
            wiener_dim=shape[1], dim=shape[0], precision=precision
        ).get_kernel(with_user_dw=False),
        arg_handler,
        is_device=True,
        n_args=5
    )
    kernel[1, 1](0., a, dt, result)
    exp_result = np.array(
        [[1.57432173e-03], [3.14864345e-03], [7.50000000e-05]]
    )
    npt.assert_allclose(
        result, exp_result, atol=tolerance[precision]
    )
