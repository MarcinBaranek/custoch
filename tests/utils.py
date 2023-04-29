import pytest
from custoch.precision import Precisions


@pytest.fixture(params=tuple(Precisions))
def precision(request) -> Precisions:
    return request.param


tolerance: dict[str, float] = {
    'float16': 1.e-2,
    'float32': 1.e-4,
    'float64': 1.e-8,
}
