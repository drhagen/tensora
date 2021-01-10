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
from typing import Generic, TypeVar


class Type:
    pass


T = TypeVar("T", bound=Type)


@dataclass(frozen=True)
class Integer(Type):
    pass


integer = Integer()


@dataclass(frozen=True)
class Float(Type):
    pass


float = Float()


@dataclass(frozen=True)
class Tensor(Type):
    pass


tensor = Tensor()


@dataclass(frozen=True)
class Mode(Type):
    pass


mode = Mode()


@dataclass(frozen=True)
class HashTable(Type):
    pass


hash_table = HashTable()


@dataclass(frozen=True)
class Pointer(Type, Generic[T]):
    target: Type


@dataclass(frozen=True)
class Array(Type, Generic[T]):
    element: Type


@dataclass(frozen=True)
class FixedArray(Type, Generic[T]):
    element: Type
    n: int
