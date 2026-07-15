from tensora import Tensor, evaluate


def assert_same_as_dense(expression, format_out, **tensor_pairs):
    tensors_in_format = {
        name: Tensor.from_lol(data, format=format) for name, (data, format) in tensor_pairs.items()
    }
    tensors_as_dense = {name: Tensor.from_lol(data) for name, (data, _) in tensor_pairs.items()}

    dense_format = "d" * (format_out.count("d") + format_out.count("s"))
    actual = evaluate(expression, format_out, **tensors_in_format)
    expected = evaluate(expression, dense_format, **tensors_as_dense)
    assert actual == expected
