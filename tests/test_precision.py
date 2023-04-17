import pytest
import numpy as np
from custoch.precision import BasePrecision


@pytest.mark.parametrize('precision', ['float16', 'float32', 'float64'])
def test_set_ok_precision(precision: str):
    obj = BasePrecision(precision=precision)
    assert obj.precision == getattr(np, precision)


@pytest.mark.parametrize('precision', ['float17', 'float31', 'float63'])
def test_set_nok_precision(precision: str):
    with pytest.raises(ValueError):
        BasePrecision(precision=precision)
