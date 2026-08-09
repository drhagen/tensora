import pytest

from tensora.ir.ast import (
    BooleanLiteral,
    FloatLiteral,
    IntegerLiteral,
    Variable,
    to_expression,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # bool must be checked before int because bool is a subclass of int
        (True, BooleanLiteral(True)),
        (False, BooleanLiteral(False)),
        (1, IntegerLiteral(1)),
        (0, IntegerLiteral(0)),
        (1.5, FloatLiteral(1.5)),
        ("x", Variable("x")),
        (Variable("y"), Variable("y")),
    ],
)
def test_to_expression(value, expected):
    assert to_expression(value) == expected
