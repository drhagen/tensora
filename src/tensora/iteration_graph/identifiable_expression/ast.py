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
]

from dataclasses import dataclass

from ...format import Mode
from .tensor_leaf import TensorLeaf


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
    variable: TensorLeaf
    indexes: tuple[str, ...]
    modes: tuple[Mode, ...]

    @property
    def name(self):
        return self.variable.name

    @property
    def order(self):
        return len(self.indexes)


@dataclass(frozen=True)
class Scalar(Variable):
    variable: TensorLeaf

    @property
    def indexes(self):
        return ()

    @property
    def modes(self):
        return ()


@dataclass(frozen=True)
class Tensor(Variable):
    variable: TensorLeaf
    indexes: tuple[str, ...]
    modes: tuple[Mode, ...]


@dataclass(frozen=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    left: Expression
    right: Expression
