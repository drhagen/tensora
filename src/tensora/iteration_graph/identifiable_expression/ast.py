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

    id: str
    name: str
    indexes: tuple[str, ...]
    modes: tuple[Mode, ...]

    @property
    def order(self):
        return len(self.indexes)


@dataclass(frozen=True, slots=True)
class Scalar(Variable):
    id: str
    name: str

    @property
    def indexes(self):
        return ()

    @property
    def modes(self):
        return ()


@dataclass(frozen=True, slots=True)
class Tensor(Variable):
    id: str
    name: str
    indexes: tuple[str, ...]
    modes: tuple[Mode, ...]


@dataclass(frozen=True, slots=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression
