from multiprocessing.pool import ThreadPool
from random import randrange

from tensora import Tensor, evaluate


def test_csr_matrix_vector_product():
    A = Tensor.from_aos([(1, 0), (0, 1), (1, 2)], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    x = Tensor.from_aos([(0,), (1,), (2,)], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    expected = Tensor.from_aos([(0,), (1,)], [-5.0, 14.0], dimensions=(2,), format="d")

    actual = evaluate("y(i) = A(i,j) * x(j)", "d", A=A, x=x)

    assert actual == expected


def test_csc_matrix_vector_product():
    A = Tensor.from_aos(
        [(1, 0), (0, 1), (1, 2)], [2.0, -2.0, 4.0], dimensions=(2, 3), format="d1s0"
    )

    x = Tensor.from_aos([(0,), (1,), (2,)], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    expected = Tensor.from_aos([(0,), (1,)], [-5.0, 14.0], dimensions=(2,), format="d")

    actual = evaluate("y(i) = A(i,j) * x(j)", "d", A=A, x=x)

    assert actual == expected


def test_csr_matrix_plus_csr_matrix():
    A = Tensor.from_aos([(1, 0), (0, 1), (1, 2)], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    B = Tensor.from_aos([(1, 1), (1, 2), (0, 2)], [-3.0, 4.0, 3.5], dimensions=(2, 3), format="ds")

    expected = Tensor.from_aos(
        [(1, 0), (0, 1), (1, 2), (1, 1), (0, 2)],
        [2.0, -2.0, 8.0, -3.0, 3.5],
        dimensions=(2, 3),
        format="ds",
    )

    actual = evaluate("C(i,j) = A(i,j) + B(i,j)", "ds", A=A, B=B)

    assert actual == expected


def test_rhs():
    A0 = Tensor.from_lol([2, -3, 0])
    A1 = Tensor.from_aos([(0, 2), (1, 2), (2, 2)], [3, 3, -3], dimensions=(3, 3), format="ds")
    A2 = Tensor.from_aos(
        [(0, 0, 1), (1, 0, 1), (2, 0, 1)], [-2, -2, 2], dimensions=(3, 3, 3), format="dss"
    )
    x = Tensor.from_lol([2, 3, 5])

    expected = Tensor.from_lol([5, 0, -3])

    assignment = "f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l) * x(k) * x(l)"

    actual = evaluate(assignment, "d", A0=A0, A1=A1, A2=A2, x=x)

    assert actual == expected


def test_many_elements_stack_overflow():
    size = 1000000  # Big enough to trigger stack overflow

    a = Tensor.from_dok({}, dimensions=(size,), format="d")
    b = evaluate("b(i) = a(i)", "d", a=a)

    assert b == Tensor.from_dok({}, dimensions=(size,), format="d")


def test_many_elements_realloc():
    size = 2000000  # Big enough to trigger realloc

    a = Tensor.from_dok({}, dimensions=(size,), format="d")
    b = evaluate("b(i) = a(i)", "s", a=a)

    assert b == Tensor.from_dok({}, dimensions=(size,), format="s")


def test_multithread_evaluation():
    # As of version 1.14.4 of cffi, the FFI.compile method is not thread safe. This tests that evaluation of different
    # kernels is thread safe.
    A = Tensor.from_aos([(1, 0), (0, 1), (1, 2)], [2.0, -2.0, 4.0], dimensions=(2, 3), format="ds")

    x = Tensor.from_aos([(0,), (1,), (2,)], [3.0, 2.5, 2.0], dimensions=(3,), format="d")

    def run_eval():
        # Generate a random expression so that the cache cannot be hit
        return evaluate(f"y{randrange(1024)}(i) = A(i,j) * x(j)", "d", A=A, x=x)

    n = 4
    with ThreadPool(n) as p:
        results = p.starmap(run_eval, [()] * n)

    expected = run_eval()

    for actual in results:
        assert actual == expected
