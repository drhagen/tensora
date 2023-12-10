__all__ = ["Expression", "Literal", "Integer", "Float", "Tensor", "Add", "Multiply"]

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


@dataclass(frozen=True, slots=True)
class Tensor(Expression):
    id: str
    name: str
    indexes: tuple[str, ...]
    modes: tuple[Mode, ...]

    @property
    def order(self):
        return len(self.indexes)


@dataclass(frozen=True, slots=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression
