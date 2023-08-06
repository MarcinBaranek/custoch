import numpy as np
import numpy.testing as npt

from custoch import KernelManager, State
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.steps import EulerStep
from custoch.euler.steps.diffusion import ClassicalEulerDiffusionStep
from ...utils import precision, tolerance
from .fixtures import *

_ = precision   # pytest fixture, just to make import warnings silent


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
            drift_function=drift_kernel,
            diffusion_step=ClassicalEulerDiffusionStep(
                function=diffusion_kernel, wiener_dim=wiener_dim,
                dim=dim, precision=precision
            )
        ).get_kernel(),
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
