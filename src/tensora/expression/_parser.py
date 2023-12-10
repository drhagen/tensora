__all__ = ["parse_assignment"]

from functools import reduce

from parsita import ParseError, ParserContext, lit, reg, rep, rep1sep, repsep
from parsita.util import splat
from returns import result

from ._exceptions import InconsistentDimensionsError, MutatingAssignmentError
from .ast import Add, Assignment, Float, Integer, Multiply, Scalar, Subtract, Tensor


def make_expression(first, rest):
    value = first
    for op, term in rest:
        match op:
            case "+":
                value = Add(value, term)
            case "-":
                value = Subtract(value, term)
    return value


class TensorExpressionParsers(ParserContext, whitespace=r"[ ]*"):
    name = reg(r"[A-Za-z][A-Za-z0-9]*")

    floating_point = reg(r"\d+((\.\d+([Ee][+-]?\d+)?)|((\.\d+)?[Ee][+-]?\d+))") > (
        lambda x: Float(float(x))
    )
    integer = reg(r"[0-9]+") > (lambda x: Integer(int(x)))
    number = floating_point | integer

    tensor = name & "(" >> repsep(name, ",") << ")" > splat(Tensor)
    scalar = name > Scalar
    variable = tensor | scalar

    parentheses = "(" >> expression << ")"  # noqa: F821
    factor = variable | number | parentheses

    term = rep1sep(factor, "*") > (lambda x: reduce(Multiply, x))
    expression = term & rep(lit("+", "-") & term) > splat(make_expression)

    assignment = variable & "=" >> expression > splat(Assignment)


def parse_assignment(
    string: str
) -> result.Result[Assignment, ParseError | MutatingAssignmentError | InconsistentDimensionsError]:
    try:
        return TensorExpressionParsers.assignment.parse(string)
    except (MutatingAssignmentError, InconsistentDimensionsError) as e:
        return result.Failure(e)
