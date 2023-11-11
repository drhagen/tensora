__all__ = ["type_to_c"]

from functools import singledispatch
from typing import Optional

from ..ir.types import *


@singledispatch
def type_to_c(type: Type, variable: Optional[str] = None):
    raise NotImplementedError(f"No implementation of type_to_c: {type}")


@type_to_c.register(Integer)
def type_to_c_integer(type: Integer, variable: Optional[str] = None):
    return "int32_t" + space_variable(variable)


@type_to_c.register(Float)
def type_to_c_float(type: Float, variable: Optional[str] = None):
    return "double" + space_variable(variable)


@type_to_c.register(Tensor)
def type_to_c_tensor(type: Tensor, variable: Optional[str] = None):
    return "taco_tensor_t" + space_variable(variable)


@type_to_c.register(Mode)
def type_to_c_mode(type: Mode, variable: Optional[str] = None):
    return "taco_mode_t" + space_variable(variable)


@type_to_c.register(HashTable)
def type_to_c_hash_table(type: HashTable, variable: Optional[str] = None):
    return "hash_table_t" + space_variable(variable)


@type_to_c.register(Pointer)
def type_to_c_pointer(type: Pointer, variable: Optional[str] = None):
    return f"{type_to_c(type.target)}* restrict" + space_variable(variable)


@type_to_c.register(Array)
def type_to_c_array(type: Array, variable: Optional[str] = None):
    return f"{type_to_c(type.element, variable)}[]"


@type_to_c.register(FixedArray)
def type_to_c_fixed_array(type: FixedArray, variable: Optional[str] = None):
    return f"{type_to_c(type.element, variable)}[{type.n}]"


def space_variable(variable: Optional[str] = None):
    if variable is None:
        return ""
    else:
        return f" {variable}"
