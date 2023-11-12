__all__ = [
    "DesugaredExpression",
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


class DesugaredExpression:
    pass


class Literal(DesugaredExpression):
    pass


@dataclass(frozen=True)
class Integer(Literal):
    value: int


@dataclass(frozen=True)
class Float(Literal):
    value: float


class Variable(DesugaredExpression):
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
class Add(DesugaredExpression):
    left: DesugaredExpression
    right: DesugaredExpression


@dataclass(frozen=True)
class Multiply(DesugaredExpression):
    left: DesugaredExpression
    right: DesugaredExpression


@dataclass(frozen=True)
class Contract(DesugaredExpression):
    index: str
    expression: DesugaredExpression


@dataclass(frozen=True)
class Assignment:
    target: Variable
    expression: DesugaredExpression
