import pytest

from tensora.compile import TensorMethod
from tensora.desugar import NoKernelFoundError
from tensora.expression import parse_assignment
from tensora.format import parse_format
from tensora.problem import make_problem

from .assert_same_as_dense import assert_same_as_dense


@pytest.mark.parametrize("dense_b", [[0, 2, 0, 3], [0, 0, 0, 0]])
@pytest.mark.parametrize("dense_c", [[5, 0, 0, 7], [0, 0, 0, 0]])
@pytest.mark.parametrize("format_b", ["s", "d"])
@pytest.mark.parametrize("format_c", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
def test_broadcast_scalar_in_product(dense_b, dense_c, format_b, format_c, format_out):
    assert_same_as_dense(
        "a(i) = b(i) * (c(i) + 1)",
        format_out,
        b=(dense_b, format_b),
        c=(dense_c, format_c),
    )


@pytest.mark.parametrize("format_b", ["dd", "ds", "sd", "ss"])
@pytest.mark.parametrize("format_c", ["s", "d"])
@pytest.mark.parametrize("format_d", ["s", "d"])
@pytest.mark.parametrize("format_out", ["dd", "ds", "sd", "ss"])
def test_different_free_indexes_in_product(format_b, format_c, format_d, format_out):
    # c(i) and d(j) broadcast over different free indexes; folding must co-iterate them without
    # duplicating b.
    assert_same_as_dense(
        "a(i,j) = b(i,j) * (c(i) + d(j))",
        format_out,
        b=([[1, 2], [3, 4]], format_b),
        c=([10, 20], format_c),
        d=([100, 200], format_d),
    )


@pytest.mark.parametrize("format_b", ["s", "d"])
@pytest.mark.parametrize("format_c", ["dd", "ds", "sd", "ss"])
@pytest.mark.parametrize("format_out", ["dd", "ds", "sd", "ss"])
def test_nested_broadcast_scalar_in_product(format_b, format_c, format_out):
    assert_same_as_dense(
        "a(i,j) = b(i) * (c(i,j) + 1)",
        format_out,
        b=([1, 2], format_b),
        c=([[3, 0], [0, 4]], format_c),
    )


@pytest.mark.parametrize("format_c", ["s", "d"])
@pytest.mark.parametrize("format_d", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
def test_product_of_sums(format_c, format_d, format_out):
    assert_same_as_dense(
        "a(i) = (c(i) + 1) * (d(i) + 1)",
        format_out,
        c=([4, 0, 6], format_c),
        d=([1, 2, 0], format_d),
    )


@pytest.mark.parametrize("format_b", ["s", "d"])
@pytest.mark.parametrize("format_c", ["dd", "ds", "sd", "ss"])
@pytest.mark.parametrize("format_d", ["s", "d"])
@pytest.mark.parametrize("format_out", ["dd", "ds", "sd", "ss"])
def test_three_term_sum_in_product(format_b, format_c, format_d, format_out):
    assert_same_as_dense(
        "a(i,j) = b(i) * (c(i,j) + d(j) + 1)",
        format_out,
        b=([1, 2], format_b),
        c=([[3, 0], [0, 4]], format_c),
        d=([5, 6], format_d),
    )


def test_contraction_in_product_is_not_yet_supported():
    # A contraction (`j`) inside the parentheses cannot be merged into a single expression because
    # a reduction is a loop, not a term. Until workspaces are implemented, no kernel exists for it.
    assignment = parse_assignment("a(i) = b(i) * (c(i) + d(j,i))").unwrap()
    formats = {
        name: parse_format(fmt).unwrap()
        for name, fmt in [("a", "d"), ("b", "d"), ("c", "d"), ("d", "dd")]
    }
    problem = make_problem(assignment, formats).unwrap()

    with pytest.raises(NoKernelFoundError):
        TensorMethod(problem)
