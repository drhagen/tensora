__all__ = ["to_identifiable"]

from functools import singledispatch

from ..format import Format
from ..iteration_graph.identifiable_expression import ast as id
from . import ast as desugar


def to_identifiable(
    assignment: desugar.Assignment,
    input_formats: dict[str, Format],
    output_format: Format,
) -> id.Assignment:
    return id.Assignment(
        to_identifiable_expression(
            assignment.target,
            {assignment.target.variable.name: output_format},
        ),
        to_identifiable_expression(assignment.expression, input_formats),
    )


@singledispatch
def to_identifiable_expression(
    expression: desugar.DesugaredExpression,
    formats: dict[str, Format],
) -> id.Expression:
    raise NotImplementedError(
        f"to_identifiable_expression not implemented for type {type(expression)}: {expression}"
    )


@to_identifiable_expression.register(desugar.Integer)
def to_identifiable_integer(expression: desugar.Integer, formats: dict[str, Format]):
    return id.Integer(expression.value)


@to_identifiable_expression.register(desugar.Float)
def to_identifiable_float(expression: desugar.Float, formats: dict[str, Format]):
    return id.Float(expression.value)


@to_identifiable_expression.register(desugar.Scalar)
def to_identifiable_scalar(expression: desugar.Scalar, formats: dict[str, Format]):
    return id.Scalar(expression.variable.to_tensor_leaf())


@to_identifiable_expression.register(desugar.Tensor)
def to_identifiable_tensor(expression: desugar.Tensor, formats: dict[str, Format]):
    return id.Tensor(
        expression.variable.to_tensor_leaf(),
        expression.indexes,
        formats[expression.variable.name].modes,
    )


@to_identifiable_expression.register(desugar.Add)
def to_identifiable_add(expression: desugar.Add, formats: dict[str, Format]):
    return id.Add(
        to_identifiable_expression(expression.left, formats),
        to_identifiable_expression(expression.right, formats),
    )


@to_identifiable_expression.register(desugar.Multiply)
def to_identifiable_multiply(expression: desugar.Multiply, formats: dict[str, Format]):
    return id.Multiply(
        to_identifiable_expression(expression.left, formats),
        to_identifiable_expression(expression.right, formats),
    )


@to_identifiable_expression.register(desugar.Contract)
def to_identifiable_contract(expression: desugar.Contract, formats: dict[str, Format]):
    return to_identifiable_expression(expression.expression, formats)
