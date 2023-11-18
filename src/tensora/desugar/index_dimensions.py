__all__ = ["index_dimensions"]

from functools import singledispatch

from ..iteration_graph import TensorLayer
from .ast import (
    Add,
    Assignment,
    Contract,
    Expression,
    Float,
    Integer,
    Multiply,
    Scalar,
    Tensor,
)


@singledispatch
def index_dimensions_expression(self: Expression) -> dict[str, TensorLayer]:
    raise NotImplementedError(f"index_dimensions not implemented for {self}")


@index_dimensions_expression.register(Integer)
@index_dimensions_expression.register(Float)
@index_dimensions_expression.register(Scalar)
def index_dimensions_nothing(self: Integer | Float | Scalar) -> dict[str, TensorLayer]:
    return {}


@index_dimensions_expression.register(Tensor)
def index_dimensions_tensor(self: Tensor) -> dict[str, TensorLayer]:
    indexes = {}
    for i, index_i in enumerate(self.indexes):
        if index_i not in indexes:
            indexes[index_i] = TensorLayer(self.variable.name, i)
    return indexes


@index_dimensions_expression.register(Add)
@index_dimensions_expression.register(Multiply)
def index_dimensions_add(self: Add | Multiply) -> dict[str, TensorLayer]:
    left_dimensions = index_dimensions_expression(self.left)
    right_dimensions = index_dimensions_expression(self.right)

    indexes = left_dimensions.copy()
    for index_i, dimension in right_dimensions.items():
        if index_i not in indexes:
            indexes[index_i] = dimension
    return indexes


@index_dimensions_expression.register(Contract)
def index_dimensions_contract(self: Contract) -> dict[str, TensorLayer]:
    return index_dimensions_expression(self.expression)


def index_dimensions(self: Assignment) -> dict[str, TensorLayer]:
    """Find a tensor name and dimension for each index in the assignment.

    The only way a kernel can know the size of an index is to get it from one
    of the tensors with a dimension indexed by that index. For each index, there
    will usually be multiple tensors whose dimension is indexed by that index,
    but they should all have the same size. This function returns the first one
    it finds for each index.
    """
    target_dimensions = index_dimensions_expression(self.target)
    right_dimensions = index_dimensions_expression(self.expression)

    indexes = target_dimensions.copy()
    for index_i, dimension in right_dimensions.items():
        if index_i not in indexes:
            indexes[index_i] = dimension
    return indexes
