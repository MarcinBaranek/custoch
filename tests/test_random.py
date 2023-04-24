import numpy as np
import numpy.testing as npt
import pytest

from custoch import KernelManager
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.random import NormalGenerator
from .utils import precision, tolerance

__all__ = ('precision', 'tolerance')


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_gen_normal_fill_array(precision, shape: tuple[int]):
    a = np.zeros(shape=shape)
    arg_handler = ArgsHandler(state=True, precision=precision)
    arg_handler.add_args(0, out=True, shape=shape, precision=precision)
    kernel = KernelManager(
        NormalGenerator.get_kernel(precision),
        arg_handler,
        is_device=True,
        n_args=2
    )
    kernel[1, 1](a)
    assert (a ** 2).sum() > 2


def test_gen_normal_match_values(precision):
    a = np.zeros(shape=(1, 4))
    arg_handler = ArgsHandler(state=True, precision=precision)
    arg_handler.add_args(0, out=True, shape=(1, 4), precision=precision)
    kernel = KernelManager(
        NormalGenerator.get_kernel(precision=precision),
        arg_handler,
        is_device=True,
        n_args=2
    )
    kernel[1, 1](a)
    npt.assert_allclose(
        a, [[0.61972869, -1.33773295, -0.44804679, -0.94795856]],
        atol=tolerance[precision]
    )
