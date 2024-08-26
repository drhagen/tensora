import pytest

from tensora import Tensor, tensor_method
from tensora.compile import BroadcastTargetIndexError
from tensora.desugar import DiagonalAccessError, NoKernelFoundError


def test_tensor_method():
    A = Tensor.from_aos([(1, 0), (0, 1), (1, 2)], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    x = Tensor.from_aos([(0,), (1,), (2,)], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    expected = Tensor.from_aos([(0,), (1,)], [-5.0, 14.0], dimensions=(2,), format="d")

    function = tensor_method("y(i) = A(i,j) * x(j)", {"y": "d", "A": "ds", "x": "d"})

    actual = function(A=A, x=x)

    assert actual == expected


def test_broadcast_target_index_error():
    with pytest.raises(BroadcastTargetIndexError):
        tensor_method("A(i,j) = a(i)", {})


def test_diagonal_error():
    with pytest.raises(DiagonalAccessError):
        tensor_method("a(i) = A(i,i)", {"a": "d", "A": "dd"})


def test_no_solution():
    with pytest.raises(NoKernelFoundError):
        tensor_method("A(i,j) = B(i,j) + C(j,i)", {"A": "ds", "B": "ds", "C": "ds"})
