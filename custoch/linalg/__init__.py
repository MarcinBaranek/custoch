# temporary solution, should be replaced
from numba import cuda
from ..typing import Vector


@cuda.jit(device=True)
def scale(a: float, x: Vector):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = a * x[i, j]


@cuda.jit(device=True)
def matmul(a, b, out):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            temp = 0
            for k in range(a.shape[1]):
                temp += a[i, k] * b[k, j]
            out[i, j] = temp


@cuda.jit(device=True)
def add(a, b, out):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            out[i, j] = a[i, j] + b[i, j]


@cuda.jit(device=True)
def write_from_to(source, destination):
    """Just copy source to destination (should have the same shape)."""
    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            destination[i, j] = source[i, j]


@cuda.jit(device=True)
def fill(matrix, value):
    """Just fill whole matrix with given value."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = value
