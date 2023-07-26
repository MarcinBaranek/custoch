from abc import ABCMeta, abstractmethod


class BaseStep(metaclass=ABCMeta):
    @abstractmethod
    def get_kernel(self, *arg, **kwargs):
        pass
