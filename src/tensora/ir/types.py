__all__ = [
    "Type",
    "Integer",
    "integer",
    "Float",
    "float",
    "Tensor",
    "tensor",
    "Mode",
    "mode",
    "HashTable",
    "hash_table",
    "Pointer",
    "Array",
    "FixedArray",
]

from dataclasses import dataclass


class Type:
    pass


@dataclass(frozen=True, slots=True)
class Integer(Type):
    pass


integer = Integer()


@dataclass(frozen=True, slots=True)
class Float(Type):
    pass


float = Float()


@dataclass(frozen=True, slots=True)
class Tensor(Type):
    pass


tensor = Tensor()


@dataclass(frozen=True, slots=True)
class Mode(Type):
    pass


mode = Mode()


@dataclass(frozen=True, slots=True)
class HashTable(Type):
    pass


hash_table = HashTable()


@dataclass(frozen=True, slots=True)
class Pointer(Type):
    target: Type


@dataclass(frozen=True, slots=True)
class Array(Type):
    element: Type


@dataclass(frozen=True, slots=True)
class FixedArray(Type):
    element: Type
    n: int
