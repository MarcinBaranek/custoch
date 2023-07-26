import numpy
import numpy as np
import numpy.testing as npt
from numba import cuda
import pytest

from custoch import KernelManager
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.euler_steps import EulerDriftStep
from ....utils import precision, tolerance

_ = precision


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
