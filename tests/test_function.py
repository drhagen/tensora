from random import randrange

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

    actual = evaluate('C(i,j) = A(i,j) + B(i,j)', 'ds', A=A, B=B)

    assert actual == expected


def test_multithread_evaluation():
    # As of version 1.14.4 of cffi, the FFI.compile method is not thread safe. This tests that evaluation of different
    # kernels is thread safe.
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

    def run_eval():
        # Generate a random expression so that the cache cannot be hit
        return evaluate(f'y{randrange(1024)}(i) = A(i,j) * x(j)', 'd', A=A, x=x)

    n = 4
    with ThreadPool(n) as p:
        results = p.starmap(run_eval, [()] * n)

    expected = run_eval()

    for actual in results:
        assert actual == expected
