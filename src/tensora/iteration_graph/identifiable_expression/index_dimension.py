__all__ = ['index_dimension']

from functools import singledispatch
from typing import Optional, Union, Tuple

from .ast import *


@singledispatch
def index_dimension(self: Node, index_variable: str) -> Optional[Tuple[str, int]]:
    pass


@index_dimension.register(Integer)
@index_dimension.register(Float)
@index_dimension.register(Scalar)
def index_dimension_nothing(self: Union[Literal, Scalar]):
    return None


@index_dimension.register(Tensor)
def index_dimension_tensor(self: Tensor, index_variable: str):
    for i, index_i in enumerate(self.indexes):
        if index_i == index_variable:
            return i
    else:
        return None


@index_dimension.register(Add)
@index_dimension.register(Subtract)
@index_dimension.register(Multiply)
def index_dimension_add(self: Union[Add, Subtract, Multiply], index_variable: str):
    left_dimension = index_dimension(self.left, index_variable)
    if left_dimension is not None:
        return left_dimension
    else:
        return index_dimension(self.right, index_variable)


@index_dimension.register(Assignment)
def index_dimension_assignment(self: Assignment, index_variable: str):
    target_dimension = index_dimension(self.target, index_variable)
    if target_dimension is not None:
        return target_dimension
    else:
        return index_dimension(self.expression, index_variable)
