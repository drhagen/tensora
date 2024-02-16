from tensora import Tensor
from tensora import evaluate_taco as evaluate


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
