import pickle

import pytest

from tensora import Format, Mode, Tensor

# The example tensors from the original taco paper:
# http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf
figure_5_taco_data = [
    ("d0d1", [[], []], [6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7]),
    ("s0d1", [[[0, 2], [0, 2]], []], [6, 0, 9, 8, 5, 0, 0, 7]),
    ("d0s1", [[], [[0, 3, 3, 5], [0, 2, 3, 0, 3]]], [6, 9, 8, 5, 7]),
    ("s0s1", [[[0, 2], [0, 2]], [[0, 3, 5], [0, 2, 3, 0, 3]]], [6, 9, 8, 5, 7]),
    ("d1d0", [[], []], [6, 0, 5, 0, 0, 0, 9, 0, 0, 8, 0, 7]),
    ("s1d0", [[[0, 3], [0, 2, 3]], []], [6, 0, 5, 9, 0, 0, 8, 0, 7]),
    ("d1s0", [[], [[0, 2, 2, 3, 5], [0, 2, 0, 0, 2]]], [6, 5, 9, 8, 7]),
    ("s1s0", [[[0, 3], [0, 2, 3]], [[0, 2, 3, 5], [0, 2, 0, 0, 2]]], [6, 5, 9, 8, 7]),
]


@pytest.mark.parametrize(("format", "indices", "vals"), figure_5_taco_data)
def test_figure_5(format, indices, vals):
    data = [[6, 0, 9, 8], [0, 0, 0, 0], [5, 0, 0, 7]]
    A = Tensor.from_lol(data, dimensions=(3, 4), format=format)

    assert A.taco_indices == indices
    assert A.taco_vals == vals


@pytest.mark.parametrize(("format", "indices", "vals"), figure_5_taco_data)
@pytest.mark.parametrize(
    "permutation",
    [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [0, 4, 3, 2, 1],
        [2, 1, 3, 4, 0],
    ],
)
def test_unsorted_coordinates(format, indices, vals, permutation):
    data = [
        ((0, 0), 6),
        ((0, 2), 9),
        ((0, 3), 8),
        ((2, 0), 5),
        ((2, 3), 7),
    ]

    permutated_data = [data[i] for i in permutation]
    coordinates, values = zip(*permutated_data, strict=False)

    A = Tensor.from_aos(coordinates, values, dimensions=(3, 4), format=format)

    assert A.taco_indices == indices
    assert A.taco_vals == vals

    A = Tensor.from_soa(zip(*coordinates, strict=False), values, dimensions=(3, 4), format=format)

    assert A.taco_indices == indices
    assert A.taco_vals == vals

    A = Tensor.from_dok(dict(permutated_data), dimensions=(3, 4), format=format)

    assert A.taco_indices == indices
    assert A.taco_vals == vals


# The example tensors from the follow-up taco paper:
# https://tensor-compiler.org/chou-oopsla18-taco-formats.pdf
@pytest.mark.parametrize(
    ("format", "indices", "vals"),
    [("d", [[]], [5, 1, 0, 0, 2, 0, 8, 0]), ("s", [[[0, 4], [0, 1, 4, 6]]], [5, 1, 2, 8])],
)
def test_figure_2a(format, indices, vals):
    data = [5, 1, 0, 0, 2, 0, 8, 0]
    a = Tensor.from_lol(data, dimensions=(8,), format=format)

    assert a.taco_indices == indices
    assert a.taco_vals == vals


@pytest.mark.parametrize(
    ("format", "indices", "vals"),
    [
        ("ds", [[], [[0, 2, 4, 4, 7], [0, 1, 0, 1, 0, 3, 4]]], [5, 1, 7, 3, 8, 4, 9]),
        (
            "ss",
            [[[0, 3], [0, 1, 3]], [[0, 2, 4, 7], [0, 1, 0, 1, 0, 3, 4]]],
            [5, 1, 7, 3, 8, 4, 9],
        ),
    ],
)
def test_figure_2e(format, indices, vals):
    data = {
        (0, 0): 5,
        (0, 1): 1,
        (1, 0): 7,
        (1, 1): 3,
        (3, 0): 8,
        (3, 3): 4,
        (3, 4): 9,
    }
    a = Tensor.from_dok(data, dimensions=(4, 6), format=format)

    assert a.taco_indices == indices
    assert a.taco_vals == vals


@pytest.mark.parametrize(
    ("format", "indices", "vals"),
    [
        (
            "sss",
            [
                [[0, 2], [0, 2]],
                [[0, 2, 5], [0, 2, 0, 2, 3]],
                [[0, 2, 3, 4, 6, 8], [0, 1, 1, 1, 0, 1, 0, 1]],
            ],
            [1, 7, 5, 2, 4, 8, 3, 9],
        ),
    ],
)
def test_figure_2m(format, indices, vals):
    data = {
        (0, 0, 0): 1,
        (0, 0, 1): 7,
        (0, 2, 1): 5,
        (2, 0, 1): 2,
        (2, 2, 0): 4,
        (2, 2, 1): 8,
        (2, 3, 0): 3,
        (2, 3, 1): 9,
    }
    a = Tensor.from_dok(data, dimensions=(3, 4, 2), format=format)

    assert a.taco_indices == indices
    assert a.taco_vals == vals


def test_from_dok():
    data = {
        (2, 2): 2.0,
        (0, 2): -3.0,
        (1, 0): 2.0,
        (2, 3): 5.0,
    }
    format = Format((Mode.compressed, Mode.compressed), (0, 1))
    x = Tensor.from_dok(data, dimensions=(4, 5), format="ss")

    assert x.order == 2
    assert x.dimensions == (4, 5)
    assert x.modes == (Mode.compressed, Mode.compressed)
    assert x.mode_ordering == (0, 1)
    assert x.format == format
    assert x.to_dok() == data


def test_from_aos():
    format = Format((Mode.dense, Mode.compressed), (1, 0))
    x = Tensor.from_aos(
        [(1, 0), (1, 1), (2, 1), (3, 1)],
        [4.5, 3.2, -3.0, 5.0],
        dimensions=(4, 3),
        format=format,
    )

    assert x.order == 2
    assert x.dimensions == (4, 3)
    assert x.modes == (Mode.dense, Mode.compressed)
    assert x.mode_ordering == (1, 0)
    assert x.format == format
    assert x.to_dok() == {
        (1, 0): 4.5,
        (1, 1): 3.2,
        (2, 1): -3.0,
        (3, 1): 5.0,
    }


def test_from_soa():
    format = Format((Mode.dense, Mode.compressed, Mode.compressed), (0, 1, 2))
    x = Tensor.from_soa(
        ([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 1]),
        [4.5, 3.2, -3.0, 5.0],
        dimensions=(2, 3, 3),
        format=format,
    )

    assert x.order == 3
    assert x.dimensions == (2, 3, 3)
    assert x.modes == (Mode.dense, Mode.compressed, Mode.compressed)
    assert x.mode_ordering == (0, 1, 2)
    assert x.format == format
    assert x.to_dok() == {
        (0, 0, 0): 4.5,
        (1, 0, 1): 3.2,
        (1, 1, 2): -3.0,
        (0, 1, 1): 5.0,
    }


def test_from_dense_lil():
    format = Format((Mode.dense, Mode.dense), (0, 1))
    x = Tensor.from_lol(
        [[0, -4.0, 4.5], [0, -3.5, 2.5]],
        dimensions=(2, 3),
        format=format,
    )

    assert x.order == 2
    assert x.dimensions == (2, 3)
    assert x.modes == (Mode.dense, Mode.dense)
    assert x.mode_ordering == (0, 1)
    assert x.format == format
    assert x.to_dok() == {
        (0, 1): -4.0,
        (0, 2): 4.5,
        (1, 1): -3.5,
        (1, 2): 2.5,
    }


def test_from_dense_lil_scalar():
    format = Format((), ())
    x = Tensor.from_lol(2.0, dimensions=(), format=format)

    assert x.order == 0
    assert x.dimensions == ()
    assert x.modes == ()
    assert x.mode_ordering == ()
    assert x.format == format
    assert x.to_dok() == {(): 2.0}


def test_to_dok():
    dok = {
        (2, 3): 2.0,
        (0, 1): 0.0,
        (1, 2): -1.0,
        (0, 3): 0.0,
    }
    dok_no_zeros = {key: value for key, value in dok.items() if value != 0}
    x = Tensor.from_dok(dok)

    assert x.to_dok() == dok_no_zeros
    assert x.to_dok(explicit_zeros=True) == dok


def test_to_float():
    x = Tensor.from_lol(2.0)
    assert float(x) == 2.0


def test_nonscalar_to_float():
    x = Tensor.from_lol([1, 2])
    with pytest.raises(ValueError, match="Can only convert Tensor of order 0 to float"):
        _ = float(x)


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        ([0, 0, 1], 3, [3, 3, 4]),
        (3, [0, 0, 1], [3, 3, 4]),
        ([1, 2, 3], [0, 0, 1], [1, 2, 4]),
    ],
)
def test_add(a, b, c):
    if isinstance(a, list):
        a = Tensor.from_lol(a)
    if isinstance(b, list):
        b = Tensor.from_lol(b)
    expected = Tensor.from_lol(c)

    actual = a + b
    assert actual == expected


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        ([0, 0, 1], 3, [-3, -3, -2]),
        (3, [0, 0, 1], [3, 3, 2]),
        ([1, 2, 3], [0, 0, 1], [1, 2, 2]),
    ],
)
def test_subtract(a, b, c):
    if isinstance(a, list):
        a = Tensor.from_lol(a)
    if isinstance(b, list):
        b = Tensor.from_lol(b)
    expected = Tensor.from_lol(c)

    actual = a - b
    assert actual == expected


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        ([0, 0, 1], 3, [0, 0, 3]),
        (3, [0, 0, 2], [0, 0, 6]),
        ([1, 2, 3], [0, 0, 4], [0, 0, 12]),
    ],
)
def test_multiply(a, b, c):
    if isinstance(a, list):
        a = Tensor.from_lol(a)
    if isinstance(b, list):
        b = Tensor.from_lol(b)
    expected = Tensor.from_lol(c)

    actual = a * b
    assert actual == expected


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        ([3, 2, 5], [1, 2, 0], 7),
        ([0, 0, 1], [[0, 0], [4, -1], [0, 2]], [0, 2]),
        ([[0, 0, 1], [2, 2, 3]], [0, -1, 2], [2, 4]),
        ([[0, 0, 1], [2, 2, 3]], [[0, 0], [4, -1], [0, 2]], [[0, 2], [8, 4]]),
    ],
)
def test_matrix_multiply(a, b, c):
    a = Tensor.from_lol(a)
    b = Tensor.from_lol(b)
    c = Tensor.from_lol(c)

    actual = a @ b
    assert actual == c


def test_binary_mismatched_dimensions():
    a = Tensor.from_lol([3, 2, 5])
    b = Tensor.from_lol([1, 2, 0, 4])

    with pytest.raises(ValueError, match="Cannot apply operator +"):
        _ = a + b

    with pytest.raises(ValueError, match="Cannot apply operator -"):
        _ = a - b

    with pytest.raises(ValueError, match="Cannot apply operator *"):
        _ = a * b


@pytest.mark.parametrize("left", [[3, 2, 5], [[0, 1, 0], [0, 2, 0]]])
@pytest.mark.parametrize("right", [[4, 1], [[0, 1, 0], [0, 2, 0]]])
def test_matrix_multiply_mismatched_dimensions(left, right):
    a = Tensor.from_lol(left)
    b = Tensor.from_lol(right)

    with pytest.raises(ValueError, match="Cannot apply operator @"):
        _ = a @ b


def test_matrix_multiply_too_many_dimensions():
    a = Tensor.from_lol([3, 2, 5])
    b = Tensor.from_dok(
        {
            (0, 0, 0): 4.5,
            (1, 0, 1): 3.2,
            (1, 1, 2): -3.0,
            (0, 1, 1): 5.0,
        },
        dimensions=(3, 3, 3),
    )

    with pytest.raises(ValueError, match="Matrix multiply is only defined"):
        _ = a @ b


def test_equality():
    assert Tensor.from_lol([0, 2, 0], format="d") == Tensor.from_lol([0, 2, 0], format="s")
    assert Tensor.from_lol([0, 1, 0], format="d") != Tensor.from_lol([0, 2, 0], format="s")
    assert Tensor.from_lol([0, 1, 0], format="d") != 1


@pytest.mark.parametrize(
    "tensor",
    [
        Tensor.from_dok({}, dimensions=()),
        Tensor.from_dok({(2, 3): 2.0, (0, 1): 0.0, (1, 2): -1.0, (0, 3): 0.0}, dimensions=(2, 4)),
        Tensor.from_dok(
            {(0, 0, 0): 4.5, (1, 0, 1): 3.2, (1, 1, 2): -3.0, (0, 1, 1): 5.0}, dimensions=(3, 3, 3)
        ),
    ],
)
def test_pickle(tensor):
    # Ensure that not only are the tensors equal, but that they also have the same format and explicit zeros, neither of
    # which affects equality
    data = pickle.dumps(tensor)
    reconstituted = pickle.loads(data)

    assert tensor == reconstituted
    assert tensor.format == reconstituted.format
    assert tensor.to_dok(explicit_zeros=True) == reconstituted.to_dok(explicit_zeros=True)


def test_str_repr():
    # Just make sure these run
    a = Tensor.from_lol([0, 2, 1, 0])
    str(a)
    repr(a)
