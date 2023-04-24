from dataclasses import dataclass, field

from typing import Optional
from numba import cuda
from numba.cuda import random


@dataclass
class State:
    n: Optional[int] = None
    seed: int = 7
    device_state: Optional[cuda.device_array] = \
        field(init=False, repr=False, default=None)

    """Class for store and create random state.

    Attributes
    ----------
    n : int
        Its number of thread used in algorithm. Optimal choose is number of 
        threads times number of blocks
    seed : int
        Random seed. default is 7.
    device_state : cuda.device_array
        Array with states stored on the GPU.
    """

    def __post_init__(self):
        if self.n:
            self.device_state = random.create_xoroshiro128p_states(
                self.n, self.seed
            )
        else:
            self.device_state = None

    def set_n(self, n: int):
        if self.n:
            raise RuntimeError('Attribute n is already set in the State!')
        self.n = n
        self.device_state = random.create_xoroshiro128p_states(
            self.n, self.seed
        )
