__all__ = ["to_identifiable"]

from functools import singledispatch

from ..format import Format
from ..iteration_graph.identifiable_expression import ast as id
from . import ast as desugar


@singledispatch
def to_identifiable(expression: desugar.Variable, formats: dict[str, Format]) -> id.Expression:
    raise NotImplementedError(
        f"to_identifiable not implemented for type {type(expression)}: {expression}"
    )


@to_identifiable.register(desugar.Scalar)
def to_identifiable_scalar(expression: desugar.Scalar, formats: dict[str, Format]):
    return id.Scalar(expression.variable.to_tensor_leaf())


@to_identifiable.register(desugar.Tensor)
def to_identifiable_tensor(expression: desugar.Tensor, formats: dict[str, Format]):
    return id.Tensor(
        expression.variable.to_tensor_leaf(),
        expression.indexes,
        formats[expression.variable.name].modes,
    )
