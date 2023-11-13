import pytest

from tensora import Tensor


@pytest.mark.parametrize(
    "array",
    [
        0.0,
        4.5,
        [],
        [[], []],
        [0, 0, 0],
        [[0, 1, 2], [0, 4, 5]],
        [[[0, 0, 3], [4, 5, 0]], [[0, 0, 0], [4, 5, 6]]],
    ],
)
@pytest.mark.parametrize("format", ["d", "s"])
def test_to_from_numpy(array, format):
    numpy = pytest.importorskip("numpy")

    expected = numpy.array(array)

    tensor = Tensor.from_numpy(expected, format=format * expected.ndim)
    actual = Tensor.to_numpy(tensor)

    assert numpy.array_equal(actual, expected)
