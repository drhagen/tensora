import pytest

from tensora import Tensor
from tensora.compile import evaluate_cffi as evaluate


def assert_same_as_dense(expression, format_out, **tensor_pairs):
    tensors_in_format = {
        name: Tensor.from_lol(data, format=format) for name, (data, format) in tensor_pairs.items()
    }
    tensors_as_dense = {name: Tensor.from_lol(data) for name, (data, _) in tensor_pairs.items()}

    dense_format = "d" * (format_out.count("d") + format_out.count("s"))
    actual = evaluate(expression, format_out, **tensors_in_format)
    expected = evaluate(expression, dense_format, **tensors_as_dense)
    assert actual == expected


@pytest.mark.parametrize("dense", [[3, 2, 4], [0, 0, 0]])
@pytest.mark.parametrize("format_in", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
def test_copy_1(dense, format_in, format_out):
    a = Tensor.from_lol(dense, format=format_in)
    actual = evaluate("b(i) = a(i)", format_out, a=a)
    assert actual == a


@pytest.mark.parametrize("dense", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("format_in", ["ss", "dd", "sd", "ds", "d1d0"])
@pytest.mark.parametrize("format_out", ["ss", "dd", "sd", "ds", "d1d0"])
def test_copy_2(dense, format_in, format_out):
    a = Tensor.from_lol(dense, format=format_in)
    actual = evaluate("b(i,j) = a(i,j)", format_out, a=a)
    assert actual == a


@pytest.mark.parametrize("dense", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("format_in", ["s1s0", "d1d0", "s1d0", "d1s0", "dd"])
@pytest.mark.parametrize("format_out", ["s1s0", "d1d0", "s1d0", "d1s0", "dd"])
def test_copy_2_backwards(dense, format_in, format_out):
    a = Tensor.from_lol(dense, format=format_in)
    actual = evaluate("b(i,j) = a(i,j)", format_out, a=a)
    assert actual == a


@pytest.mark.parametrize("expression", [0, 1])
def test_constant_scalar(expression):
    actual = evaluate(f"a() = {expression}", "")
    assert actual == Tensor.from_lol(expression)


@pytest.mark.parametrize("dense1", [[0, 2, 4, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("format1", ["s", "d"])
@pytest.mark.parametrize("format2", ["s", "d"])
def test_vector_dot(dense1, dense2, format1, format2):
    assert_same_as_dense(
        "out() = in1(i) * in2(i)", "", in1=(dense1, format1), in2=(dense2, format2)
    )


@pytest.mark.parametrize("dense1", [[0, 2, 4, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("format1", ["s", "d"])
@pytest.mark.parametrize("format2", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
@pytest.mark.parametrize("operator", ["+", "-", "*"])
def test_vector_binary(operator, dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        f"out(i) = in1(i) {operator} in2(i)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
    )


@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[[-1, 3.5], [0, 0], [4, 0]], [[0, 0], [0, 0], [0, 0]]])
@pytest.mark.parametrize("format1", ["ss", "dd", "sd", "ds", "d1d0"])
@pytest.mark.parametrize("format2", ["ss", "dd", "sd", "ds", "d1d0"])
@pytest.mark.parametrize("format_out", ["dd", "d1d0"])
def test_matrix_dot(dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        "out(i,k) = in1(i,j) * in2(j,k)", format_out, in1=(dense1, format1), in2=(dense2, format2)
    )


@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0], [0, 0, 0]])
@pytest.mark.parametrize("format1", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format2", ["s", "d"])
@pytest.mark.parametrize("format_out", ["d"])
def test_matrix_vector_product(dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        "out(i) = in1(i,j) * in2(j)", format_out, in1=(dense1, format1), in2=(dense2, format2)
    )


@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[[-1, 3.5], [0, 0], [4, 0]], [[0, 0], [0, 0], [0, 0]]])
@pytest.mark.parametrize("dense3", [[[-3, 0], [7, 0]], [[0, 0], [0, 0]]])
@pytest.mark.parametrize("format1", ["dd", "ds"])
@pytest.mark.parametrize("format2", ["dd"])
@pytest.mark.parametrize("format3", ["dd", "ds"])
@pytest.mark.parametrize("format_out", ["dd"])
def test_matrix_multiply_add(dense1, dense2, dense3, format1, format2, format3, format_out):
    assert_same_as_dense(
        "out(i,k) = in1(i,j) * in2(j,k) + in3(i,k)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
        in3=(dense3, format3),
    )


@pytest.mark.parametrize(
    "dense_b",
    [
        [
            [[0, 2, 4, 0], [0, -1, 0, 3], [1, -1, 0, 0]],
            [[-2, 4, 0, 0], [0, 0, 0, 3], [1, 1, 0, 0]],
        ],
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
    ],
)
@pytest.mark.parametrize(
    "dense_d",
    [
        [[-1, 3.5, 1, 2, 0], [0, 2, 6, 3, 0], [4, 0, 0, 1, -1], [0, 0, 3, 6, 9]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ],
)
@pytest.mark.parametrize(
    "dense_c",
    [
        [[0, 0, 1, 2, 7], [7, 0, 5, 2, 0], [-1, 0, 0, 2, 1]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ],
)
@pytest.mark.parametrize("format_b", ["ddd", "dss", "sss", "ssd", "d1d0s2", "s0d2d1"])
@pytest.mark.parametrize("format_d", ["dd", "ds", "ss"])
@pytest.mark.parametrize("format_c", ["dd", "ds", "ss"])
@pytest.mark.parametrize("format_out", ["dd", "d1d0"])
def test_mttkrp(dense_b, dense_d, dense_c, format_b, format_d, format_c, format_out):
    assert_same_as_dense(
        "A(i,j) = B(i,k,l) * D(l,j) * C(k,j)",
        format_out,
        B=(dense_b, format_b),
        D=(dense_d, format_d),
        C=(dense_c, format_c),
    )
