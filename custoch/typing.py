from typing import TypeVar, Generic

__all__ = (
    'd', 'm', 'One', 'RandomState', 'Vector', 'Matrix', 'Out', 'dW', 'Time'
)

d = TypeVar('d', bound=int)
"""Dimension of the space variable."""
m = TypeVar('m', bound=int)
"""Dimension of the wiener process."""
One = TypeVar('One', bound=int)
RandomState = TypeVar('RandomState')
dW = TypeVar('dW')
T = TypeVar('T')
S = TypeVar('S')
Time = float
"""Generally timestamp used in function describing a equations."""


class FirstDim(Generic[T]):
    pass


class SecondDim(Generic[S]):
    pass


class Vector(FirstDim[T], SecondDim[S]):
    pass


class Matrix(FirstDim[T], SecondDim[S]):
    pass


class Out(FirstDim[T], SecondDim[S]):
    pass
