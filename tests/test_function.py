import pytest
from itertools import repeat
from multiprocessing.pool import ThreadPool
from tensora import Tensor, tensor_method, evaluate


def test_csr_matrix_vector_product():
    A = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]],
        [2.0, -2.0, 4.0],
        dimensions=(2, 3), format='ds'
    )

    x = Tensor.from_aos(
        [[0], [1], [2]],
        [3.0, 2.5, 2.0],
        dimensions=(3,), format='d'
    )

    expected = Tensor.from_aos(
        [[0], [1]],
        [-5.0, 14.0],
        dimensions=(2,), format='d'
    )

    function = tensor_method('y(i) = A(i,j) * x(j)', dict(A='ds', x='d'), 'd')

    actual = function(A, x)

    assert actual == expected

    actual = evaluate('y(i) = A(i,j) * x(j)', 'd', A=A, x=x)

    assert actual == expected


def test_csc_matrix_vector_product():
    A = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]],
        [2.0, -2.0, 4.0],
        dimensions=(2, 3), format='d1s0'
    )

    x = Tensor.from_aos(
        [[0], [1], [2]],
        [3.0, 2.5, 2.0],
        dimensions=(3,), format='d'
    )

    expected = Tensor.from_aos(
        [[0], [1]],
        [-5.0, 14.0],
        dimensions=(2,), format='d'
    )

    function = tensor_method('y(i) = A(i,j) * x(j)', dict(A='d1s0', x='d'), 'd')

    actual = function(A, x)

    assert actual == expected

    actual = evaluate('y(i) = A(i,j) * x(j)', 'd', A=A, x=x)

    assert actual == expected


@pytest.mark.skip('taco fails on sparse outputs')
def test_csr_matrix_plus_csr_matrix():
    A = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]],
        [2.0, -2.0, 4.0],
        dimensions=(2, 3), format='ds'
    )

    B = Tensor.from_aos(
        [[1, 1], [1, 2], [0, 2]],
        [-3.0, 4.0, 3.5],
        dimensions=(2, 3), format='ds'
    )

    expected = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2], [1, 1], [0, 2]],
        [2.0, -2.0, 8.0, -3.0, 3.5],
        dimensions=(2, 3), format='ds'
    )

    function = tensor_method('C(i,j) = A(i,j) + B(i,j)', dict(A='ds', B='ds'), 'ds')

    actual = function(A, B)

    assert actual == expected

    actual = evaluate('C(i,j) = A(i,j) * B(i,j)', 'ds', A=A, B=B)

    assert actual == expected


def test_multithread_evaluation():
    # simply check if this is run. without lock we will have a race condition
    def run_eval(args):
        A, x = args
        return evaluate('y(i) = A(i,j) * x(j)', 'd', A=A, x=x)

    A = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]],
        [2.0, -2.0, 4.0],
        dimensions=(2, 3), format='ds'
    )

    x = Tensor.from_aos(
        [[0], [1], [2]],
        [3.0, 2.5, 2.0],
        dimensions=(3,), format='d'
    )

    all_inputs = zip(repeat(A, 2), repeat(x, 2))

    with ThreadPool(2) as p:
        p.map(run_eval, all_inputs)
