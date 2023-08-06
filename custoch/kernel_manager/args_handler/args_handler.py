from __future__ import annotations

from typing import Any, Optional, Iterable

import numpy as np

from .array_handler import ArrayHandler
from custoch.precision import BasePrecision
from custoch.state import State


class ArgsHandler(BasePrecision):
    """Class responsible for handling all arguments of given CUDA kernel.

    To handle arguments that are array like, the `ArrayHandler` is using.
    """
    array_handlers: dict[int, ArrayHandler]
    """Dictionary with `ArrayHandler` for parameter in position respectively 
    to keys of the dictionary."""
    to_out: list[int]
    """For arguments with positional index in `to_out` the content will be 
    copied to host after call kernel"""
    state: Optional[State]
    """Random state for Kernel."""
    args: list[Any]

    def __init__(
            self,
            state: Optional[State | bool] = None,
            precision: str = 'float64',
            to_out: Optional[Iterable[int]] = None
    ):
        super().__init__(precision=precision)
        if state is True:
            state = State()
        self.state = state
        self.array_handlers = {}
        self.to_out = [] if to_out is None else list(to_out)

    @staticmethod
    def create_from_args(*args, state=False) -> ArgsHandler:
        """Create `ArgsHandler` in the fly.

        On the runtime the function needs to know if kernel is using random
        state or not.

        This method could not work correctly in all cases.
        Should be used only in really simple cases.
        """
        obj = ArgsHandler(state=state)
        for idx, arg in enumerate(args):
            try:
                arr = np.array(arg)
            except Exception:
                continue
            obj.add_args(
                idx, out=True, shape=arr.shape, precision=str(arr.dtype)
            )
        return obj

    def validate_configuration(self) -> None:
        for index in self.to_out:
            assert index in self.array_handlers.keys(),\
                f'For argument with results form device, ' \
                f'an `ArrayHandler` should be created! Check index: {index}'

    def add_args(
            self,
            index: int,
            out: bool = False,
            shape: Optional[tuple[int, ...]] = None,
            precision: Optional[str] = None,
            name: Optional[str] = None
    ) -> None:
        """Add information about parameter to `ArgsHandler`.

        Parameters
        ----------
        index: int
            Positional index of parameter
        out: bool
            Indicate if after call kernel the content of this argument should
            be copied to the host.
            Default False.
        shape: Optional[tuple[int, ...]]
            Optional expected shape of the parameter if is array like.
            Default None.
        precision: Optional[str]
            Optional precision of the argument.
        name: Optional[str]
            For debug purpose. If given argument don't met validation
            condition then, the name will be added to the error message.

        Returns
        -------
        None

        Raises
        ------
        TypeError: When `index` is not int.
        ValueError: When `index` is negative.
        RuntimeError: When parameter with given `index` is already added.
        """
        if not isinstance(index, int):
            raise TypeError(
                f'index should be int, got: {index} with type: {type(index)}.'
            )
        if index < 0:
            raise ValueError(f'Index should be not negative, got: {index}.')
        if index in self.array_handlers:
            raise RuntimeError(
                f'Argument with index {index} is already added!'
            )
        if out:
            self.to_out.append(index)
        self.array_handlers[index] = ArrayHandler(
            None,
            shape=shape,
            precision=precision if precision else self.precision,
            name=name
        )

    def prepare_state(self, grid: Optional[tuple[int, int]] = None) -> None:
        if len(grid) != 2:
            raise NotImplementedError(
                'Grid with dimension differ form 2 is not Supported!'
            )
        if grid is None:
            return
        if self.state:
            self.state.n = grid[0] * grid[1]

    def __call__(
            self, *args, grid: Optional[tuple[int, int]] = None
    ) -> list[Any]:
        self.validate_configuration()
        self.prepare_state(grid)
        self.args = list(args)
        device_args = []
        for index, arg in enumerate(self.args):
            if index in self.array_handlers:
                self.array_handlers[index].array = arg
                device_args.append(self.array_handlers[index].to_device())
            else:
                device_args.append(arg)
        if self.state:
            device_args.append(self.state.device_state)
        return device_args

    def copy_to_host(self) -> None:
        for index in self.to_out:
            self.args[index][:] = self.array_handlers[index].to_host()[:]
