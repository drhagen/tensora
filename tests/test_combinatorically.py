import pytest

from tensora import Tensor, evaluate_taco, evaluate_tensora


def assert_same_as_dense(expression, format_out, evaluate, **tensor_pairs):
    tensors_in_format = {
        name: Tensor.from_lol(data, format=format) for name, (data, format) in tensor_pairs.items()
    }
    tensors_as_dense = {name: Tensor.from_lol(data) for name, (data, _) in tensor_pairs.items()}

    actual = evaluate(expression, format_out, **tensors_in_format)
    expected = evaluate(expression, "d" * len(format_out), **tensors_as_dense)
    assert actual == expected


pytestmark = pytest.mark.parametrize("evaluate", [evaluate_taco, evaluate_tensora])


@pytest.mark.parametrize("dense", [[3, 2, 4], [0, 0, 0]])
@pytest.mark.parametrize("format_in", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
def test_copy_1(evaluate, dense, format_in, format_out):
    a = Tensor.from_lol(dense, format=format_in)
    actual = evaluate("b(i) = a(i)", format_out, a=a)
    assert actual == a


@pytest.mark.skip("taco fails to compile most of these")
@pytest.mark.parametrize("dense", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("format_in", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format_out", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
def test_copy_2(evaluate, dense, format_in, format_out):
    a = Tensor.from_lol(dense, format=format_in)
    actual = evaluate("b(i,j) = a(i,j)", format_out, a=a)
    assert actual == a


@pytest.mark.skip("taco fails on all of these")
@pytest.mark.parametrize(
    "dense", [[[0, 2, 4], [0, -1, 0], [2, 0, 3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
)
@pytest.mark.parametrize("format_in", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format_out", ["s", "d"])
def test_diag(evaluate, dense, format_in, format_out):
    assert_same_as_dense("diagA(i) = A(i,i)", format_out, A=(dense, format_in), evaluate=evaluate)


@pytest.mark.parametrize("dense1", [[0, 2, 4, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("format1", ["s", "d"])
@pytest.mark.parametrize("format2", ["s", "d"])
def test_vector_dot(evaluate, dense1, dense2, format1, format2):
    assert_same_as_dense(
        "out = in1(i) * in2(i)",
        "",
        in1=(dense1, format1),
        in2=(dense2, format2),
        evaluate=evaluate,
    )


@pytest.mark.parametrize("dense1", [[0, 2, 4, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("format1", ["s", "d"])
@pytest.mark.parametrize("format2", ["s", "d"])
@pytest.mark.parametrize("format_out", ["s", "d"])
@pytest.mark.parametrize("operator", ["+", "-", "*"])
def test_vector_binary(evaluate, operator, dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        f"out(i) = in1(i) {operator} in2(i)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
        evaluate=evaluate,
    )


@pytest.mark.skip("taco fails to compile most of these")
@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[[-1, 3.5], [0, 0], [4, 0]], [[0, 0], [0, 0], [0, 0]]])
@pytest.mark.parametrize("format1", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format2", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format_out", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
def test_matrix_dot(evaluate, dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        "out(i,k) = in1(i,j) * in2(j,k)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
        evaluate=evaluate,
    )


@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[-1, 3.5, 0], [0, 0, 0]])
@pytest.mark.parametrize("format1", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format2", ["s", "d"])
@pytest.mark.parametrize("format_out", ["d"])
def test_matrix_vector_product(evaluate, dense1, dense2, format1, format2, format_out):
    assert_same_as_dense(
        "out(i) = in1(i,j) * in2(j)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
        evaluate=evaluate,
    )


@pytest.mark.parametrize("dense1", [[[0, 2, 4], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
@pytest.mark.parametrize("dense2", [[[-1, 3.5], [0, 0], [4, 0]], [[0, 0], [0, 0], [0, 0]]])
@pytest.mark.parametrize("dense3", [[[-3, 0], [7, 0]], [[0, 0], [0, 0]]])
@pytest.mark.parametrize("format1", ["dd", "ds"])
@pytest.mark.parametrize("format2", ["dd"])
@pytest.mark.parametrize("format3", ["dd", "ds"])
@pytest.mark.parametrize("format_out", ["dd"])
def test_matrix_multiply_add(
    evaluate, dense1, dense2, dense3, format1, format2, format3, format_out
):
    assert_same_as_dense(
        "out(i,k) = in1(i,j) * in2(j,k) + in3(i,k)",
        format_out,
        in1=(dense1, format1),
        in2=(dense2, format2),
        in3=(dense3, format3),
        evaluate=evaluate,
    )
