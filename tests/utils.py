import pytest
from custoch.precision import Precisions


@pytest.fixture(params=tuple(Precisions))
def precision(request):
    return request.param


tolerance = {
    'float16': 1.e-2,
    'float32': 1.e-4,
    'float64': 1.e-8,
}
