import numpy as np

from custoch import KernelManager, State
from custoch.kernel_manager.args_handler import ArgsHandler
from custoch.precision import Precisions
from custoch.euler.euler_steps import EulerPath
from ..utils import precision, tolerance
from .test_steps.fixtures import *

__all__ = ('precision', 'tolerance')


# @pytest.mark.parametrize('dt', [0.01, 0.0001, 0.1, 1, 10])
@pytest.mark.parametrize('shape', [(3, 2), (2, 3), (6, 4)])
@pytest.mark.parametrize('diffusion_kernel', [1, 2, 3, 4], indirect=True)
def test_euler_path(
        precision: Precisions, diffusion_kernel, shape: tuple[int]
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
