from numba import cuda


class ArgHandler:
    default_precision: str
    __device_args: dict[int, cuda.device_array]

