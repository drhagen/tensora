__all__ = [
    "Expression",
    "Literal",
    "Integer",
    "Float",
    "Tensor",
    "Add",
    "Multiply",
    "Contract",
    "Assignment",
]

from dataclasses import dataclass


class Expression:
    __slots__ = ()


class Literal(Expression):
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Integer(Literal):
    value: int


@dataclass(frozen=True, slots=True)
class Float(Literal):
    value: float


@dataclass(frozen=True, slots=True)
class Tensor(Expression):
    id: int
    name: str
    indexes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Contract(Expression):
    index: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class Assignment:
    target: Tensor
    expression: Expression
