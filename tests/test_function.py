from multiprocessing.pool import ThreadPool
from random import randrange

import pytest

from tensora import Tensor, evaluate_taco, evaluate_tensora, tensor_method
from tensora.desugar import DiagonalAccessError, NoKernelFoundError
from tensora.generate import TensorCompiler

pytestmark = pytest.mark.parametrize(
    ("evaluate", "compiler"),
    [(evaluate_taco, TensorCompiler.taco), (evaluate_tensora, TensorCompiler.tensora)],
)


def test_csr_matrix_vector_product(evaluate, compiler):
    A = Tensor.from_aos([[1, 0], [0, 1], [1, 2]], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    x = Tensor.from_aos([[0], [1], [2]], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    expected = Tensor.from_aos([[0], [1]], [-5.0, 14.0], dimensions=(2,), format="d")

    function = tensor_method("y(i) = A(i,j) * x(j)", dict(A="ds"), compiler=compiler)

    actual = function(A=A, x=x)

    assert actual == expected

    actual = evaluate("y(i) = A(i,j) * x(j)", "d", A=A, x=x)

    assert actual == expected


def test_csc_matrix_vector_product(evaluate, compiler):
    A = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2]], [2.0, -2.0, 4.0], dimensions=(2, 3), format="d1s0"
    )

    x = Tensor.from_aos([[0], [1], [2]], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    expected = Tensor.from_aos([[0], [1]], [-5.0, 14.0], dimensions=(2,), format="d")

    function = tensor_method("y(i) = A(i,j) * x(j)", dict(A="d1s0"), compiler=compiler)

    actual = function(A=A, x=x)

    assert actual == expected

    actual = evaluate("y(i) = A(i,j) * x(j)", "d", A=A, x=x)

    assert actual == expected


def test_csr_matrix_plus_csr_matrix(evaluate, compiler):
    A = Tensor.from_aos([[1, 0], [0, 1], [1, 2]], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    B = Tensor.from_aos([[1, 1], [1, 2], [0, 2]], [-3.0, 4.0, 3.5], dimensions=(2, 3), format="ds")

    expected = Tensor.from_aos(
        [[1, 0], [0, 1], [1, 2], [1, 1], [0, 2]],
        [2.0, -2.0, 8.0, -3.0, 3.5],
        dimensions=(2, 3),
        format="ds",
    )

    function = tensor_method(
        "C(i,j) = A(i,j) + B(i,j)", dict(C="ds", A="ds", B="ds"), compiler=compiler
    )

    actual = function(A=A, B=B)

    assert actual == expected

    actual = evaluate("C(i,j) = A(i,j) + B(i,j)", "ds", A=A, B=B)

    assert actual == expected


def test_rhs(evaluate, compiler):
    if compiler == TensorCompiler.taco:
        pytest.skip("Taco does not support this")

    A0 = Tensor.from_lol([2, -3, 0])
    A1 = Tensor.from_aos([(0, 2), (1, 2), (2, 2)], [3, 3, -3], dimensions=(3, 3), format="ds")
    A2 = Tensor.from_aos(
        [(0, 0, 1), (1, 0, 1), (2, 0, 1)], [-2, -2, 2], dimensions=(3, 3, 3), format="dss"
    )
    x = Tensor.from_lol([2, 3, 5])

    expected = Tensor.from_lol([5, 0, -3])

    assignment = "f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l) * x(k) * x(l)"
    formats = {"f": "d", "A0": "d", "A1": "ds", "A2": "dss", "x": "d"}

    function = tensor_method(assignment, formats, compiler=compiler)

    actual = function(A0=A0, A1=A1, A2=A2, x=x)

    assert actual == expected

    actual = evaluate(assignment, "d", A0=A0, A1=A1, A2=A2, x=x)

    assert actual == expected


def test_multithread_evaluation(evaluate, compiler):
    # As of version 1.14.4 of cffi, the FFI.compile method is not thread safe. This tests that evaluation of different
    # kernels is thread safe.
    A = Tensor.from_aos([[1, 0], [0, 1], [1, 2]], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    x = Tensor.from_aos([[0], [1], [2]], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    def run_eval():
        # Generate a random expression so that the cache cannot be hit
        return evaluate(f"y{randrange(1024)}(i) = A(i,j) * x(j)", "d", A=A, x=x)

    n = 4
    with ThreadPool(n) as p:
        results = p.starmap(run_eval, [()] * n)

    expected = run_eval()

    for actual in results:
        assert actual == expected


def test_diagonal_error(evaluate, compiler):
    if compiler == TensorCompiler.taco:
        pytest.skip("We do not currently get a nice error from Taco")

    with pytest.raises(DiagonalAccessError):
        tensor_method("a(i) = A(i,i)", dict(a="d", A="dd"), compiler)


def test_no_solution(evaluate, compiler):
    if compiler == TensorCompiler.taco:
        pytest.xfail("Taco currently segfaults on this")

    with pytest.raises(NoKernelFoundError):
        tensor_method("A(i,j) = B(i,j) + C(j,i)", dict(A="ds", B="ds", C="ds"), compiler)
