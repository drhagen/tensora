import pytest

from tensora import Tensor


@pytest.fixture
def a_ds():
    return Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds"
    )


@pytest.fixture
def b_ds():
    return Tensor.from_aos(
        [[1, 1], [1, 2], [0, 2]], [-3.0, 4.0, 3.5], dimensions=(2, 3), format="ds"
    )


@pytest.fixture
def c_ds_add():
    return Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2], [1, 1], [0, 2]],
        [2.0, -2.0, 8.0, -3.0, 3.5],
        dimensions=(2, 3),
        format="ds",
    )


def test_add_ds_ds(a_ds, b_ds, c_ds_add):
    actual = a_ds + b_ds

    assert actual == c_ds_add
