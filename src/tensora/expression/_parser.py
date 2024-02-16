__all__ = ["parse_assignment"]

from functools import reduce

from parsita import ParseError, ParserContext, lit, reg, rep, rep1sep, repsep
from parsita.util import splat
from returns import result

from ._exceptions import InconsistentDimensionsError, MutatingAssignmentError, NameConflictError
from .ast import Add, Assignment, Float, Integer, Multiply, Subtract, Tensor


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

    tensor = name & "(" >> (repsep(name, ",") > tuple) << ")" > splat(Tensor)

    parentheses = "(" >> expression << ")"  # noqa: F821
    factor = tensor | number | parentheses

    term = rep1sep(factor, "*") > (lambda x: reduce(Multiply, x))
    expression = term & rep(lit("+", "-") & term) > splat(make_expression)

    assignment = tensor & "=" >> expression > splat(Assignment)


def parse_assignment(
    string: str,
) -> result.Result[
    Assignment,
    ParseError | MutatingAssignmentError | InconsistentDimensionsError | NameConflictError,
]:
    try:
        return TensorExpressionParsers.assignment.parse(string)
    except (MutatingAssignmentError, InconsistentDimensionsError, NameConflictError) as e:
        return result.Failure(e)
