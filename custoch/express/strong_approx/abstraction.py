from typing import Callable
from abc import ABCMeta

from ...typing import *
from ...precision import Precisions
from .config import StrongApproxConfig


class WeakApproxAbstraction(StrongApproxConfig, metaclass=ABCMeta):
    """Class for weak approximation of the SDE."""
    def __init__(
        self,
        drift_function: Callable[[Time, Vector[d, One], Out[d, One]], None],
        diffusion_kernel: Callable[[Time, Vector[d, One], Out[d, m]], None],
        wiener_dim: m = None,
        dim: d = None,
        precision: Precisions = None,
        start_of_time: t_0 = None,
        end_of_time: T = None,
        grid_size: N = None,
        number_of_trajectories: int = None
    ) -> None:
        for arg_name in self.fields:
            if locals().get(arg_name, None) is not None:
                setattr(self, arg_name, locals().get(arg_name))
