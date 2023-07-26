from numba import cuda

from .typing import Matrix


@cuda.jit(device=True)
def sse(array_a: Matrix, array_b: Matrix):
    total = 0
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            total += (array_a[i, j] - array_b[i, j]) ** 2
    return total
