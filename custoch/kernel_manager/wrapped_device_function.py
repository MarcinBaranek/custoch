from numba import cuda


__all__ = ('get_device_function_wrapper',)


def get_device_function_wrapper(device_function, n_args: int):
    """Wrap device kernel in to a global kernel.

    Parameters
    ----------
    device_function: Device CUDA kernel, wrapped by numba
    n_args: int
        Number of all arguments taken by device kernel.
    """
    def n_arg_wrapper_1(func):
        @cuda.jit
        def wrapped_device_function(a):
            func(a)

        return wrapped_device_function

    def n_arg_wrapper_2(func):
        @cuda.jit
        def wrapped_device_function(a, b):
            func(a, b)

        return wrapped_device_function

    def n_arg_wrapper_3(func):
        @cuda.jit
        def wrapped_device_function(a, b, c):
            func(a, b, c)

        return wrapped_device_function

    def n_arg_wrapper_4(func):
        @cuda.jit
        def wrapped_device_function(a, b, c, d):
            func(a, b, c, d)

        return wrapped_device_function

    def n_arg_wrapper_5(func):
        @cuda.jit
        def wrapped_device_function(a, b, c, d, e):
            func(a, b, c, d, e)

        return wrapped_device_function

    def n_arg_wrapper_6(func):
        @cuda.jit
        def wrapped_device_function(a, b, c, d, e, f):
            func(a, b, c, d, e, f)

        return wrapped_device_function

    def n_arg_wrapper_7(func):
        @cuda.jit
        def wrapped_device_function(a, b, c, d, e, f, g):
            func(a, b, c, d, e, f, g)

        return wrapped_device_function

    def n_arg_wrapper_8(func):
        @cuda.jit
        def wrapped_device_function(a, b, c, d, e, f, g, h):
            func(a, b, c, d, e, f, g, h)

        return wrapped_device_function

    if n_args not in range(1, 9):
        raise ValueError(
            f'Wrapper for device function with {n_args} is not supported. '
            f'To obtain support for this number of the argument '
            f'please contact with the creators'
        )
    return locals()[f'n_arg_wrapper_{n_args}'](device_function)
