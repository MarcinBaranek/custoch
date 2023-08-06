import pytest
import numpy as np
from custoch.precision import BasePrecision, Precisions


@pytest.mark.parametrize('precision', ['float16', 'float32', 'float64'])
def test_set_ok_precision(precision: str):
    obj = BasePrecision(precision=precision)
    assert obj.precision == getattr(np, precision)


@pytest.mark.parametrize('precision', ['float17', 'float31', 'float63'])
def test_set_nok_precision(precision: str):
    with pytest.raises(ValueError):
        BasePrecision(precision=precision)


@pytest.mark.parametrize('precision', [Precisions.float16, Precisions.float64])
def test_set_custoch_precision(precision: str):
    obj = BasePrecision(precision=precision)
    assert obj.precision == getattr(np, precision)


@pytest.mark.parametrize('precision', [np.float32, np.float64])
def test_set_numpy_precision(precision: str):
    obj = BasePrecision(precision=precision)
    assert obj.precision == precision
