__all__ = [
    "Expression",
    "Literal",
    "Integer",
    "Float",
    "Variable",
    "Scalar",
    "Tensor",
    "Add",
    "Multiply",
    "Contract",
    "Assignment",
]

from dataclasses import dataclass
from typing import List

from .id import Id


class Expression:
    pass


class Literal(Expression):
    pass


@dataclass(frozen=True)
class Integer(Literal):
    value: int


@dataclass(frozen=True)
class Float(Literal):
    value: float


class Variable(Expression):
    variable: Id
    indexes: List[str]


@dataclass(frozen=True)
class Scalar(Variable):
    variable: Id

    @property
    def indexes(self):
        return []


@dataclass(frozen=True)
class Tensor(Variable):
    variable: Id
    indexes: List[str]


@dataclass(frozen=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Contract(Expression):
    index: str
    expression: Expression


@dataclass(frozen=True)
class Assignment:
    target: Variable
    expression: Expression
