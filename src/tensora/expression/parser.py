__all__ = ['parse_assignment']

from functools import reduce

from parsita import TextParsers, lit, reg, rep, rep1sep, Result
from parsita.util import splat

from .ast import Assignment, Add, Subtract, Multiply, Tensor, Scalar, Integer, Float


def make_expression(first, rest):
    value = first
    for op, term in rest:
        if op == '+':
            value = Add(value, term)
        else:
            value = Subtract(value, term)
    return value


class TensorExpressionParsers(TextParsers):
    name = reg(r'[A-Za-z][A-Za-z0-9]*')

    # taco does not support negatives or exponents
    floating_point = reg(r'[0-9]+\.[0-9]+') > (lambda x: Float(float(x)))
    integer = reg(r'[0-9]+') > (lambda x: Integer(int(x)))
    number = floating_point | integer

    # taco requires at least one index; scalar tensors are not parsed as `a()`
    # taco also allows for `y_{i}` and `y_i` to mean `y(i)`, but that is not supported here
    tensor = name & '(' >> rep1sep(name, ',') << ')' > splat(Tensor)
    scalar = name > Scalar
    variable = tensor | scalar

    parentheses = '(' >> expression << ')'  # noqa: F821
    factor = variable | number | parentheses

    term = rep1sep(factor, '*') > (lambda x: reduce(Multiply, x))
    expression = term & rep(lit('+', '-') & term) > splat(make_expression)

    simple_assignment = variable & '=' >> expression > splat(Assignment)
    add_assignment = variable & '+=' >> expression > splat(lambda v, e: Assignment(v, Add(v, e)))

    assignment = simple_assignment | add_assignment


def parse_assignment(string: str) -> Result[Assignment]:
    return TensorExpressionParsers.assignment.parse(string)
