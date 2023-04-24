from custoch.kernel_manager.args_handler.array_handler import ArrayHandler
import pytest


class TestShapeValidations:

    def test_type_error(self):
        with pytest.raises(TypeError) as err:
            ArrayHandler(shape=range(2))
        assert err.match('Shape should be instance of tuple got: *')

    def test_type_list_is_ok(self):
        ArrayHandler(shape=[1, 2])

    def test_to_big_len(self):
        with pytest.raises(AssertionError) as err:
            ArrayHandler(shape=(1, 2, 3))
        assert err.match('Maximal allowed shape\'s dimension is 2, got shap *')

    def test_to_short_len(self):
        with pytest.raises(AssertionError) as err:
            ArrayHandler(shape=tuple())
        assert err.match('Minimal allowed shape\'s dimension is 1, got shap *')

    def test_negative_values(self):
        with pytest.raises(AssertionError) as err:
            ArrayHandler(shape=(1, -2))
        assert err.match('Shape should have positive coefficients.')


class TestArray:

    def test_incorrect_shape(self):
        with pytest.raises(AssertionError) as err:
            ArrayHandler(shape=(1,), array=[[1, 2], [3, 4]])
        assert err.match('Array shape should be *')

    def test_send_arr_to_device(self):
        with pytest.raises(RuntimeError) as err:
            arr = ArrayHandler(array=None)
            arr.to_device()
        assert err.match(
            'Array is still None and could not be sent to the device!'
        )

    def test_get_arr_from_device(self):
        with pytest.raises(RuntimeError) as err:
            arr = ArrayHandler(array=None)
            arr.to_host()
        assert err.match('No data was sent to the device!')
