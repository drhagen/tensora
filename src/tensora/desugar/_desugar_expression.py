__all__ = ["desugar_assignment"]

from functools import singledispatch
from itertools import count
from typing import Iterator

from ..expression import ast as sugar
from . import ast as desugar


@singledispatch
def desugar_expression(
    self: sugar.Expression, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    raise NotImplementedError(f"desugar_expression not implemented for {type(self)}: {self}")


@desugar_expression.register(sugar.Integer)
def desugar_integer(
    self: sugar.Integer, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    return desugar.Integer(self.value)


@desugar_expression.register(sugar.Float)
def desugar_float(
    self: sugar.Float, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    return desugar.Float(self.value)


@desugar_expression.register(sugar.Tensor)
def desugar_tensor(
    self: sugar.Tensor, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    output = desugar.Tensor(next(ids), self.name, self.indexes)
    for index in contract_indexes:
        output = desugar.Contract(index, output)
    return output


@desugar_expression.register(sugar.Add)
def desugar_add(
    self: sugar.Add, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(self.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(self.right.index_participants().keys()).intersection(contract_indexes)

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Add(
        desugar_expression(self.left, left_indexes - intersection_indexes, ids),
        desugar_expression(self.right, right_indexes - intersection_indexes, ids),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


@desugar_expression.register(sugar.Subtract)
def desugar_subtract(
    self: sugar.Subtract, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(self.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(self.right.index_participants().keys()).intersection(contract_indexes)

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Add(
        desugar_expression(self.left, left_indexes - intersection_indexes, ids),
        desugar.Multiply(
            desugar.Integer(-1),
            desugar_expression(self.right, right_indexes - intersection_indexes, ids),
        ),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


@desugar_expression.register(sugar.Multiply)
def desugar_multiply(
    self: sugar.Multiply, contract_indexes: set[str], ids: Iterator[int]
) -> desugar.Expression:
    left_indexes = set(self.left.index_participants().keys()).intersection(contract_indexes)
    right_indexes = set(self.right.index_participants().keys()).intersection(contract_indexes)

    intersection_indexes = left_indexes.intersection(right_indexes)

    output = desugar.Multiply(
        desugar_expression(self.left, left_indexes - intersection_indexes, ids),
        desugar_expression(self.right, right_indexes - intersection_indexes, ids),
    )

    for index in intersection_indexes:
        output = desugar.Contract(index, output)

    return output


def desugar_assignment(assignment: sugar.Assignment) -> desugar.Assignment:
    ids = count()

    desugared_target = desugar.Tensor(next(ids), assignment.target.name, assignment.target.indexes)

    all_indexes = set(assignment.index_participants().keys())
    contract_indexes = all_indexes - set(assignment.target.indexes)

    desugared_right_hand_side = desugar_expression(assignment.expression, contract_indexes, ids)

    return desugar.Assignment(desugared_target, desugared_right_hand_side)
