import pytest

from tensora import Format, Mode, Tensor

try:
    # numpy is an optional dependency of tensora. It is only used to convert from numpy.
    import numpy
except ImportError:  # pragma: no cover
    numpy = None

try:
    # scipy is an optional dependency of tensora. It is only used to convert from scipy.sparse.
    import scipy.sparse as scipy_sparse
except ImportError:  # pragma: no cover
    scipy_sparse = None


def test_from_dok():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    actual = Tensor.from_dok(dok, dimensions=(3, 3, 2), format="sss")

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.compressed, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (3, 3, 2)


def test_from_dok_default_dimensions_default_format():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    actual_dok = Tensor.from_dok(dok)

    assert actual_dok.to_dok() == dok
    assert actual_dok.format == Format((Mode.dense, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual_dok.dimensions == (2, 3, 2)


def test_from_soa():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    soa = ([1, 0, 1, 1], [2, 2, 0, 2], [1, 1, 0, 0])

    values = [
        4.5,
        2.0,
        -2.0,
        4.5,
    ]

    actual = Tensor.from_soa(soa, values, dimensions=(3, 3, 2), format="sss")

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.compressed, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (3, 3, 2)


def test_from_soa_default_dimensions_default_format():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    soa = ([1, 0, 1, 1], [2, 2, 0, 2], [1, 1, 0, 0])

    values = [
        4.5,
        2.0,
        -2.0,
        4.5,
    ]

    actual = Tensor.from_soa(soa, values)

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.dense, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (2, 3, 2)


def test_from_aos():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    aos = [
        (1, 2, 1),
        (0, 2, 1),
        (1, 0, 0),
        (1, 2, 0),
    ]

    values = [
        4.5,
        2.0,
        -2.0,
        4.5,
    ]

    actual = Tensor.from_aos(aos, values, dimensions=(3, 3, 2), format="sss")

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.compressed, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (3, 3, 2)


def test_from_aos_default_dimensions_default_format():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    aos = [
        (1, 2, 1),
        (0, 2, 1),
        (1, 0, 0),
        (1, 2, 0),
    ]

    values = [
        4.5,
        2.0,
        -2.0,
        4.5,
    ]

    actual = Tensor.from_aos(aos, values)

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.dense, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (2, 3, 2)


def test_from_lol():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    lol = [[[0, 0], [0, 0], [0, 2]], [[-2, 0], [0, 0], [4.5, 4.5]], [[0, 0], [0, 0], [0, 0]]]

    actual = Tensor.from_lol(lol, dimensions=(3, 3, 2), format="sss")

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.compressed, Mode.compressed, Mode.compressed), (0, 1, 2))
    assert actual.dimensions == (3, 3, 2)


def test_from_lol_default_dimensions_default_format():
    dok = {
        (1, 2, 1): 4.5,
        (0, 2, 1): 2.0,
        (1, 0, 0): -2.0,
        (1, 2, 0): 4.5,
    }

    lol = [[[0, 0], [0, 0], [0, 2]], [[-2, 0], [0, 0], [4.5, 4.5]], [[0, 0], [0, 0], [0, 0]]]

    actual = Tensor.from_lol(lol)

    assert actual.to_dok() == dok
    assert actual.format == Format((Mode.dense, Mode.dense, Mode.dense), (0, 1, 2))
    assert actual.dimensions == (3, 3, 2)


def test_from_lol_scalar():
    actual = Tensor.from_lol(3.5)

    assert actual.to_dok() == {(): 3.5}
    assert actual.format == Format((), ())
    assert actual.dimensions == ()


def assert_all_methods_same(lol, dimensions, format):
    dok = {}

    def recurse(tree, indexes):
        if isinstance(tree, (int, float)):
            dok[indexes] = tree
        else:
            for i_element, element in enumerate(tree):
                recurse(element, (*indexes, i_element))

    recurse(lol, ())

    dok_no_zeros = {key: value for key, value in dok.items() if value != 0}
    coordinates = list(dok.keys())
    values = list(dok.values())
    coordinates_no_zeros = list(dok_no_zeros.keys())
    values_no_zeros = list(dok_no_zeros.values())

    tensor_from_lol = Tensor.from_lol(lol, dimensions=dimensions, format=format)
    tensor_from_dok = Tensor.from_dok(dok, dimensions=dimensions, format=format)
    tensor_from_dok_no_zeros = Tensor.from_dok(dok_no_zeros, dimensions=dimensions, format=format)
    tensor_from_aos = Tensor.from_aos(coordinates, values, dimensions=dimensions, format=format)
    tensor_from_aos_no_zeros = Tensor.from_aos(
        coordinates_no_zeros, values_no_zeros, dimensions=dimensions, format=format
    )

    assert tensor_from_dok == tensor_from_lol
    assert tensor_from_dok_no_zeros == tensor_from_lol
    assert tensor_from_aos == tensor_from_lol
    assert tensor_from_aos_no_zeros == tensor_from_lol

    if numpy is not None:
        # Apply reshape so that zero dimensions appear correctly
        numpy_array = numpy.asarray(lol).reshape(dimensions)
        tensor_from_numpy = Tensor.from_numpy(numpy_array, format=format)

        assert tensor_from_numpy == tensor_from_lol

    if scipy_sparse is not None and len(dimensions) == 2:
        # Scipy sparse only allows matrices
        if len(values) == 0:
            # zip only transposes if the iterable is not empty
            soa = ([], [])
        else:
            soa = tuple(zip(*coordinates, strict=False))
        scipy_csc_matrix = scipy_sparse.csc_matrix((values, soa), shape=dimensions)
        scipy_csr_matrix = scipy_sparse.csr_matrix((values, soa), shape=dimensions)
        scipy_coo_matrix = scipy_sparse.coo_matrix((values, soa), shape=dimensions)

        assert Tensor.from_scipy_sparse(scipy_csc_matrix) == tensor_from_lol
        assert Tensor.from_scipy_sparse(scipy_csr_matrix) == tensor_from_lol
        assert Tensor.from_scipy_sparse(scipy_coo_matrix) == tensor_from_lol


@pytest.mark.parametrize("lol", [3.5, 0.0, 1])
def test_convert_0(lol):
    assert_all_methods_same(lol, (), "")


@pytest.mark.parametrize(
    ("lol", "dimensions"),
    [
        ([], (0,)),
        ([3], (1,)),
        ([3, -2, 0], (3,)),
    ],
)
@pytest.mark.parametrize("format", ["d", "s"])
def test_convert_1(lol, dimensions, format):
    assert_all_methods_same(lol, dimensions, format)


@pytest.mark.parametrize(
    ("lol", "dimensions"),
    [
        ([], (0, 0)),
        ([], (0, 3)),
        ([[], []], (2, 0)),
        ([[0, -2, 3]], (1, 3)),
        ([[0], [3]], (2, 1)),
        ([[0, 2, 3], [0, -2, 3]], (2, 3)),
        ([[0, 0, 0], [0, 0, 0]], (2, 3)),
    ],
)
@pytest.mark.parametrize("format", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
def test_convert_2(lol, dimensions, format):
    assert_all_methods_same(lol, dimensions, format)


@pytest.mark.parametrize(
    ("lol", "dimensions"),
    [
        ([], (0, 0, 0)),
        ([], (0, 2, 3)),
        ([[], [], [], []], (4, 0, 3)),
        ([[[], []], [[], []], [[], []], [[], []]], (4, 2, 0)),
        ([[[0, 0, 0], [1, 2, 3]]], (1, 2, 3)),
        ([[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 1]], [[1, 0, 0]]], (4, 1, 3)),
        ([[[0], [1]], [[0], [-2]], [[0], [0]], [[0], [1]]], (4, 2, 1)),
        (
            [
                [[0, 0, 0], [1, 2, 3]],
                [[0, 0, 0], [-2, 3, 0]],
                [[0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [1, 0, 0]],
            ],
            (4, 2, 3),
        ),
        (
            [
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            (4, 2, 3),
        ),
    ],
)
@pytest.mark.parametrize("modes", ["sss", "ssd", "sds", "sdd", "dss", "dsd", "dds", "ddd"])
@pytest.mark.parametrize("ordering", ["012", "021", "120", "102", "201", "210"])
def test_convert_3(lol, dimensions, modes, ordering):
    format = modes[0] + ordering[0] + modes[1] + ordering[1] + modes[2] + ordering[2]
    assert_all_methods_same(lol, dimensions, format)


@pytest.mark.parametrize(
    "data",
    [
        0,
        2.0,
    ],
)
def test_interconversion_0(data):
    expected = Tensor.from_lol(data)
    actual = expected.to_format("")

    assert expected == actual


@pytest.mark.parametrize(
    "data",
    [
        [0, 0],
        [0, 1],
    ],
)
@pytest.mark.parametrize("format1", ["d", "s"])
@pytest.mark.parametrize("format2", ["d", "s"])
def test_interconversion_1(data, format1, format2):
    expected = Tensor.from_lol(data)
    actual = expected.to_format(format2)

    assert expected == actual


@pytest.mark.parametrize(
    "data",
    [
        [[0, 0], [0, 0]],
        [[0, 1], [-2, 0]],
    ],
)
@pytest.mark.parametrize("format1", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
@pytest.mark.parametrize("format2", ["ss", "dd", "sd", "ds", "s1s0", "d1d0", "s1d0", "d1s0"])
def test_interconversion_2(data, format1, format2):
    expected = Tensor.from_lol(data)
    actual = expected.to_format(format2)

    assert expected == actual
