from typing import TypeVar, Generic

__all__ = (
    'd', 'm', 'One', 'RandomState', 'Vector', 'Matrix', 'Out', 'dW', 'Time',
    'T', 't_0', 'N'
)

d = TypeVar('d', bound=int)
"""Dimension of the space variable."""

m = TypeVar('m', bound=int)
"""Dimension of the wiener process."""

One = TypeVar('One', bound=int)
"""Denote 1 in shapes.

Examples
--------
>>> def func(point: Vector[One, d]) -> None:
...     pass
"""
RandomState = TypeVar('RandomState')
dW = TypeVar('dW')
"""Typing increments of the Wiener process"""

T = TypeVar('T', bound=float)
"""Variable to typing end of the time interval considering equation."""

N = TypeVar('N', bound=int)
t_0 = TypeVar('t_0', bound=float)
"""Variable to typing start of the time interval considering equation."""

S = TypeVar('S')
Time = float
"""Generally timestamp used in function describing a equations."""


class FirstDim(Generic[T]):
    pass


class SecondDim(Generic[S]):
    pass


class Vector(FirstDim[T], SecondDim[S]):
    """Class for typing Vector.

    Normally on of the typed dimension is `One`.

    Examples
    --------
    >>> def func(vector: Vector[One, d]) -> None:
    ...     pass
    """
    shape: tuple[int, int]

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass


class Matrix(FirstDim[T], SecondDim[S]):
    """Class for typing matrices."""
    shape: tuple[int, int]

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass


class Out(FirstDim[T], SecondDim[S]):
    """Typing array changing by the function.
    As cuda kernel are able to return only a number,
    we introduce the `Out` as typing for arguments that will be over writen
    by given function. This pattern allows to "return" much complex objects.
    Currently, is used for matrices with the shape: `FirstDim, SecondDim`

    Examples
    --------
    >>> def func(point: Vector[One, d], result: Out[d, One]) -> None:
    ...     pass
    """
    pass
