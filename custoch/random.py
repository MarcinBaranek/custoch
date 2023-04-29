from typing import Callable
from abc import ABCMeta, abstractmethod

from numba import cuda
from numba.core.types import float16
from numba.cuda.random import (
    xoroshiro128p_normal_float32, xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64
)

from .precision import Precisions

__all__ = ('RandomGenerator', 'UniformGenerator', 'NormalGenerator')


class RandomGenerator(metaclass=ABCMeta):
    """Abstract class for random generators."""
    @staticmethod
    @abstractmethod
    def get_kernel(
            precision: Precisions
    ) -> Callable[[float, float, cuda.device_array], float] \
            | Callable[[cuda.device_array, cuda.device_array], None]:
        """Return CUDA kernel for generating random numbers."""
        pass


class UniformGenerator(RandomGenerator):
    @staticmethod
    def get_kernel(
            precision
    ) -> Callable[[float, float, cuda.device_array], float]:
        match precision:
            case 'float16':
                return UniformGenerator.get_for_float16()
            case 'float32':
                return UniformGenerator.get_for_float32()
            case 'float64':
                return UniformGenerator.get_for_float64()
            case _:
                raise ValueError(
                    f'Given precision: {precision} is not supported. '
                    f'Use one of the {tuple(map(str, Precisions))}.'
                )

    @staticmethod
    def get_for_float16() \
            -> Callable[[float, float, cuda.device_array], float]:
        @cuda.jit(device=True)
        def __gen_uniform_16(start, end, state):
            thread_id = cuda.grid(1)
            return start + (end - start) * float16(
                xoroshiro128p_uniform_float32(state, thread_id)
            )

        return __gen_uniform_16

    @staticmethod
    def get_for_float32() \
            -> Callable[[float, float, cuda.device_array], float]:
        @cuda.jit(device=True)
        def __gen_uniform_32(start, end, state):
            thread_id = cuda.grid(1)
            return start + (end - start) * \
                   xoroshiro128p_uniform_float32(state, thread_id)

        return __gen_uniform_32

    @staticmethod
    def get_for_float64() \
            -> Callable[[float, float, cuda.device_array], float]:
        @cuda.jit(device=True)
        def __gen_uniform_64(start, end, state):
            thread_id = cuda.grid(1)
            return start + (end - start) * \
                   xoroshiro128p_uniform_float64(state, thread_id)

        return __gen_uniform_64


class NormalGenerator(RandomGenerator):
    @staticmethod
    def get_kernel(
            precision
    ) -> Callable[[cuda.device_array, cuda.device_array], None]:
        match precision:
            case 'float16':
                return NormalGenerator.get_for_float16()
            case 'float32':
                return NormalGenerator.get_for_float32()
            case 'float64':
                return NormalGenerator.get_for_float64()
            case _:
                raise ValueError(
                    f'Given precision: {precision} is not supported. '
                    f'Use one of the {tuple(map(str, Precisions))}.'
                )

    @staticmethod
    def get_for_float16() \
            -> Callable[[cuda.device_array, cuda.device_array], None]:
        @cuda.jit(device=True)
        def __gen_normal_16(place_holder, state):
            thread_id = cuda.grid(1)
            for i in range(place_holder.shape[0]):
                for j in range(place_holder.shape[1]):
                    place_holder[i, j] = \
                        float16(xoroshiro128p_normal_float32(state, thread_id))

        return __gen_normal_16

    @staticmethod
    def get_for_float32() \
            -> Callable[[cuda.device_array, cuda.device_array], None]:
        @cuda.jit(device=True)
        def __gen_normal_32(place_holder, state):
            thread_id = cuda.grid(1)
            for i in range(place_holder.shape[0]):
                for j in range(place_holder.shape[1]):
                    place_holder[i, j] = \
                        xoroshiro128p_normal_float32(state, thread_id)

        return __gen_normal_32

    @staticmethod
    def get_for_float64() \
            -> Callable[[cuda.device_array, cuda.device_array], None]:
        @cuda.jit(device=True)
        def __gen_normal_64(place_holder, state):
            thread_id = cuda.grid(1)
            for i in range(place_holder.shape[0]):
                for j in range(place_holder.shape[1]):
                    place_holder[i, j] = \
                        xoroshiro128p_normal_float64(state, thread_id)

        return __gen_normal_64
