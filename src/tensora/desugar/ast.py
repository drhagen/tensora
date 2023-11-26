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


class Variable(Expression):
    __slots__ = ()

    id: int
    name: str
    indexes: list[str]


@dataclass(frozen=True, slots=True)
class Scalar(Variable):
    id: int
    name: str

    @property
    def indexes(self):
        return []


@dataclass(frozen=True, slots=True)
class Tensor(Variable):
    id: int
    name: str
    indexes: list[str]


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
    target: Variable
    expression: Expression
