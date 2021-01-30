from tensora import Tensor


try:
    # numpy is an optional dependency of tensora. It is only used to convert from numpy.
    import numpy
except ImportError:  # pragma: no cover
    numpy = None

if numpy is not None:
    def test_to_numpy():
        expected = numpy.array([[[0, 0, 3], [4, 5, 0]], [[0, 0, 0], [4, 5, 6]]])

        tensor = Tensor.from_numpy(expected, format='sss')
        actual = Tensor.to_numpy(tensor)

        assert numpy.array_equal(actual, expected)
