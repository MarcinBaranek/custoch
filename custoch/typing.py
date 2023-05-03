from typing import TypeVar, Generic

__all__ = ('d', 'm', 'One', 'RandomState', 'Vector', 'Matrix', 'Out', 'dW')
d = TypeVar('d', bound=int)
"""Dimension of the space variable."""
m = TypeVar('m', bound=int)
"""Dimension of the wiener process."""
One = TypeVar('One', bound=int)
RandomState = TypeVar('RandomState')
dW = TypeVar('dW')
T = TypeVar('T')


class Vector(Generic[T]):
    pass


class Matrix(Generic[T]):
    pass


class Out(Generic[T]):
    pass
