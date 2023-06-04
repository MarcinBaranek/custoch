import pytest

from custoch.state import State


@pytest.mark.parametrize('n', [None, 1, 20, 10, 13])
@pytest.mark.parametrize('seed', [1, 20, 10, 134444])
def test_str(n, seed):
    state = State(n, seed)
    assert str(state) == f'State(n: {n}, seed: {seed})'


def test_validation():
    with pytest.raises(ValueError):
        State(-1)
    with pytest.raises(TypeError):
        State(10.)
