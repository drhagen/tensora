__all__ = ['index_dimensions']

from functools import singledispatch
from typing import Union, Dict, Tuple

from .ast import *


@singledispatch
def index_dimensions(self: Node) -> Dict[str, Tuple[str, int]]:
    pass


@index_dimensions.register(Integer)
@index_dimensions.register(Float)
@index_dimensions.register(Scalar)
def index_dimensions_nothing(self: Union[Literal, Scalar]):
    return {}


@index_dimensions.register(Tensor)
def index_dimensions_tensor(self: Tensor):
    indexes = {}
    for i, index_i in enumerate(self.indexes):
        if index_i not in indexes:
            indexes[index_i] = (self.name, i)
    return indexes


@index_dimensions.register(Add)
@index_dimensions.register(Subtract)
@index_dimensions.register(Multiply)
def index_dimensions_add(self: Union[Add, Subtract, Multiply]):
    left_dimensions = index_dimensions(self.left)
    right_dimensions = index_dimensions(self.right)
    indexes = left_dimensions.copy()
    for index_i, dimension in right_dimensions.items():
        if index_i not in indexes:
            indexes[index_i] = dimension
    return indexes


@index_dimensions.register(Assignment)
def index_dimensions_assignment(self: Assignment):
    target_dimensions = index_dimensions(self.target)
    right_dimensions = index_dimensions(self.expression)
    indexes = target_dimensions.copy()
    for index_i, dimension in right_dimensions.items():
        if index_i not in indexes:
            indexes[index_i] = dimension
    return indexes
