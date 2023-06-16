import pytest
from parsita import Failure

from tensora import Format, Mode
from tensora.format import parse_format

format_strings = [
    ('', Format((), ())),
    ('d', Format((Mode.dense,), (0,))),
    ('s', Format((Mode.compressed,), (0,))),
    ('ds', Format((Mode.dense, Mode.compressed), (0, 1))),
    ('sd', Format((Mode.compressed, Mode.dense), (0, 1))),
    ('d1s0', Format((Mode.dense, Mode.compressed), (1, 0))),
    ('d1s0s2', Format((Mode.dense, Mode.compressed, Mode.compressed), (1, 0, 2))),
]


@pytest.mark.parametrize('string,format', format_strings)
def test_parse_format(string, format):
    actual = parse_format(string).unwrap()
    assert actual == format


@pytest.mark.parametrize('string,format', format_strings)
def test_deparse_format(string, format):
    actual = format.deparse()
    assert actual == string


@pytest.mark.parametrize('string', ['df', '1d0s', 'd0s', 'd0s1s1', 'd1s2s3'])
def test_parse_bad_format(string):
    actual = parse_format(string)
    assert isinstance(actual, Failure)


def test_format_attributes():
    format = Format((Mode.dense, Mode.compressed), (1, 0))

    assert format.order == 2
    assert format.modes[0] == Mode.dense


def test_mode_dense_attributes():
    mode_dense = Mode.from_c_int(0)
    assert mode_dense.c_int == 0
    assert mode_dense.character == 'd'


def test_mode_sparse_attributes():
    mode_dense = Mode.from_c_int(1)
    assert mode_dense.c_int == 1
    assert mode_dense.character == 's'


def test_mode_from_illegal_int():
    with pytest.raises(ValueError):
        Mode.from_c_int(3)


def test_differing_sizes():
    with pytest.raises(ValueError):
        _ = Format((Mode.dense,), (0, 1))
