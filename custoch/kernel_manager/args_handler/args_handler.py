from __future__ import annotations

from typing import Any, Optional, Iterable

import numpy as np

from .array_handler import ArrayHandler
from custoch.precision import BasePrecision
from custoch.state import State


class ArgsHandler(BasePrecision):
    array_handlers: dict[int, ArrayHandler]
    to_out: list[int]
    state: Optional[State]
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
                'For argument with results form device, ' \
                'an `ArrayHandler` should be created!'

    def add_args(
            self,
            index: int,
            out: bool = False,
            shape: Optional[tuple[int, ...]] = None,
            precision: Optional[str] = None,
    ) -> None:
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
            precision=precision if precision else str(self.precision)
        )

    def prepare_state(self, grid: Optional[tuple[int, int]] = None) -> None:
        if grid is None:
            return
        if self.state and self.state.n is None:
            self.state.set_n(grid[0] * grid[1])

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
