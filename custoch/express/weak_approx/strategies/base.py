from abc import ABCMeta, abstractmethod

from ..abstraction import WeakApproxAbstraction


class AbstractWeakApproxStrategy(metaclass=ABCMeta):
    @abstractmethod
    def get_internal_kernel(self, weak_approx_config: WeakApproxAbstraction):
        pass

    @abstractmethod
    def get_kernel(self, weak_approx_config: WeakApproxAbstraction):
        pass

    @abstractmethod
    def get_state_shape(self, weak_approx_config: WeakApproxAbstraction):
        pass
