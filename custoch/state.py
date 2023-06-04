from typing import Optional
from numba import cuda
from numba.cuda import random


class State:
    n: Optional[int]
    seed: int
    device_state: Optional[cuda.device_array]

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

    def __init__(self, n: Optional[int] = None, seed: int = 7) -> None:
        self._n = n
        self.seed = seed
        if n:
            self.device_state = random.create_xoroshiro128p_states(
                self.n, self.seed
            )
        else:
            self.device_state = None

    @property
    def n(self) -> int:
        if self._n is None:
            raise RuntimeError(f'The value of n in {self} is no set yet!')
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f'n should be int, got: {value} with type: {type(value)}!'
            )
        if value <= 0:
            raise ValueError(f'n should be positive int, got: {value}!')
        self._n = value
        if value is not None:
            self.device_state = random.create_xoroshiro128p_states(
                self.n, self.seed
            )

    def __str__(self) -> str:
        return f'State(n: {self._n}, seed: {self.seed})'
