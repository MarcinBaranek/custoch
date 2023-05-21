# temporary solution, should be replaced
from numba import cuda


@cuda.jit(device=True)
def scale(a, x):
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
