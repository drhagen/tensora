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
    "Pointer",
    "Array",
    "FixedArray",
]

from dataclasses import dataclass


class Type:
    __slots__ = ()


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
class Pointer(Type):
    target: Type


@dataclass(frozen=True, slots=True)
class Array(Type):
    element: Type


@dataclass(frozen=True, slots=True)
class FixedArray(Type):
    element: Type
    n: int
