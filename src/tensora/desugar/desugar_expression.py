__all__ = ["desugar_assignment"]

from functools import singledispatch
from itertools import count
from typing import Iterator, Set

from ..expression import ast as sugar
from . import ast as desugar
from . import id


@singledispatch
def desugar_expression(
    expression: sugar.Expression, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    raise NotImplementedError(
        f"desugar_expression not implemented for {type(expression)}: {expression}"
    )


@desugar_expression.register(sugar.Integer)
def desugar_integer(
    expression: sugar.Integer, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    return desugar.Integer(expression.value)


@desugar_expression.register(sugar.Float)
def desugar_float(
    expression: sugar.Float, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    return desugar.Float(expression.value)


@desugar_expression.register(sugar.Scalar)
def desugar_scalar(
    expression: sugar.Scalar, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    return desugar.Scalar(id.Id(expression.name, next(ids)))


@desugar_expression.register(sugar.Tensor)
def desugar_tensor(
    expression: sugar.Tensor, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    output = desugar.Tensor(id.Id(expression.name, next(ids)), expression.indexes)
    for index in contract_indexes:
        output = desugar.Contract(index, output)
    return output


@desugar_expression.register(sugar.Add)
def desugar_add(
    expression: sugar.Add, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(expression.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(expression.right.index_participants().keys()).intersection(
        contract_indexes
    )

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Add(
        desugar_expression(expression.left, left_indexes - intersection_indexes, ids),
        desugar_expression(expression.right, right_indexes - intersection_indexes, ids),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


@desugar_expression.register(sugar.Subtract)
def desugar_subtract(
    expression: sugar.Subtract, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(expression.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(expression.right.index_participants().keys()).intersection(
        contract_indexes
    )

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Add(
        desugar_expression(expression.left, left_indexes - intersection_indexes, ids),
        desugar.Multiply(
            desugar.Integer(-1),
            desugar_expression(expression.right, right_indexes - intersection_indexes, ids),
        ),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


@desugar_expression.register(sugar.Multiply)
def desugar_multiply(
    expression: sugar.Multiply, contract_indexes: Set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(expression.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(expression.right.index_participants().keys()).intersection(
        contract_indexes
    )

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Multiply(
        desugar_expression(expression.left, left_indexes - intersection_indexes, ids),
        desugar_expression(expression.right, right_indexes - intersection_indexes, ids),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


def desugar_assignment(assignment: sugar.Assignment) -> desugar.Assignment:
    if isinstance(assignment.target, sugar.Scalar):
        desugared_target = desugar.Scalar(id.Id(assignment.target.name, 0))
    elif isinstance(assignment.target, sugar.Tensor):
        desugared_target = desugar.Tensor(
            id.Id(assignment.target.name, 0), assignment.target.indexes
        )
    else:
        raise NotImplementedError

    all_indexes = set(assignment.index_participants().keys())
    contract_indexes = all_indexes - set(assignment.target.indexes)

    desugared_right_hand_side = desugar_expression(
        assignment.expression, contract_indexes, count(1)
    )

    return desugar.Assignment(desugared_target, desugared_right_hand_side)
