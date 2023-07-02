from .base import AbstractWeakApproxStrategy
from .only_last import WeakApproxOnlyLastStrategy
from .whole_trajectories import WeakApproxWholeTrajectoriesStrategy

__all__ = (
    'WeakApproxOnlyLastStrategy', 'WeakApproxWholeTrajectoriesStrategy',
    'AbstractWeakApproxStrategy'
)
