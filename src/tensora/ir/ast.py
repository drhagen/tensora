from __future__ import annotations

__all__ = [
    "Statement",
    "Expression",
    "Assignable",
    "Variable",
    "AttributeAccess",
    "ArrayIndex",
    "IntegerLiteral",
    "FloatLiteral",
    "BooleanLiteral",
    "ModeLiteral",
    "ArrayLiteral",
    "Add",
    "Subtract",
    "Multiply",
    "Equal",
    "NotEqual",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
    "And",
    "Or",
    "FunctionCall",
    "Max",
    "Min",
    "Address",
    "BooleanToInteger",
    "Allocate",
    "ArrayAllocate",
    "ArrayReallocate",
    "Free",
    "Declaration",
    "Assignment",
    "DeclarationAssignment",
    "Block",
    "Branch",
    "Loop",
    "Break",
    "Return",
    "FunctionDefinition",
]

from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Union

from ..format import Mode
from .types import Type


class Statement:
    pass


class Expression(Statement):
    def plus(self, term: Union[Expression, int, str]):
        term = to_expression(term)
        return Add(self, term)

    def minus(self, term: Union[Expression, int, str]):
        term = to_expression(term)
        return Subtract(self, term)

    def times(self, factor: Union[Expression, int, str]):
        factor = to_expression(factor)
        return Multiply(self, factor)


class Assignable(Expression):
    def attr(self, attribute: str):
        return AttributeAccess(self, attribute)

    def idx(self, index: Union[Expression, int, str]):
        index = to_expression(index)
        return ArrayIndex(self, index)

    def assign(self, value: Union[Expression, int, str]):
        value = to_expression(value)
        return Assignment(self, value)

    def increment(self, amount: Union[Expression, int, str] = 1):
        amount = to_expression(amount)
        return self.assign(self.plus(amount))


@dataclass(frozen=True)
class Variable(Assignable):
    __slots__ = ("name",)
    name: str

    def declare(self, type: Type):
        return Declaration(self, type)


@dataclass(frozen=True)
class AttributeAccess(Assignable):
    __slots__ = ("target", "attribute")
    target: Assignable
    attribute: str


@dataclass(frozen=True)
class ArrayIndex(Assignable):
    __slots__ = ("target", "index")
    target: Assignable
    index: Expression


@dataclass(frozen=True)
class IntegerLiteral(Expression):
    __slots__ = ("value",)
    value: int


@dataclass(frozen=True)
class FloatLiteral(Expression):
    __slots__ = ("value",)
    value: float


@dataclass(frozen=True)
class BooleanLiteral(Expression):
    __slots__ = ("value",)
    value: bool


@dataclass(frozen=True)
class ModeLiteral(Expression):
    value: Mode


@dataclass(frozen=True)
class ArrayLiteral(Expression):
    __slots__ = ("value",)
    elements: list[Expression]


@dataclass(frozen=True)
class Add(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    @staticmethod
    def join(terms: List[Union[Expression, int, str]]):
        terms = map(to_expression, terms)
        return reduce(And, terms, IntegerLiteral(0))


@dataclass(frozen=True)
class Subtract(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    def __init__(self, left: Expression, right: Expression):
        if left is None:
            assert left is not None
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @staticmethod
    def join(factors: List[Union[Expression, int, str]]):
        factors = map(to_expression, factors)
        return reduce(Multiply, factors, IntegerLiteral(1))


@dataclass(frozen=True)
class Equal(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class NotEqual(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class GreaterThan(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class LessThan(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class GreaterThanOrEqual(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class LessThanOrEqual(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


@dataclass(frozen=True)
class And(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: List[Union[Expression, int, str]]):
        operands = map(to_expression, operands)
        return reduce(And, operands, BooleanLiteral(True))


@dataclass(frozen=True)
class Or(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: List[Union[Expression, int, str]]):
        operands = map(to_expression, operands)
        return reduce(Or, operands, BooleanLiteral(False))


@dataclass(frozen=True)
class FunctionCall(Expression):
    __slots__ = ("name", "arguments")
    name: Variable
    arguments: list[Expression]


@dataclass(frozen=True)
class Max(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: List[Union[Expression, int, str]]):
        operands = map(to_expression, operands)
        return reduce(Max, operands)


@dataclass(frozen=True)
class Min(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: List[Union[Expression, int, str]]):
        operands = map(to_expression, operands)
        return reduce(Min, operands)


@dataclass(frozen=True)
class Address(Expression):
    __slots__ = ("target",)
    target: Variable


@dataclass(frozen=True)
class BooleanToInteger(Expression):
    __slots__ = ("expression",)
    expression: Expression


@dataclass(frozen=True)
class Allocate(Expression):
    __slots__ = ("type",)
    type: Type


@dataclass(frozen=True)
class ArrayAllocate(Expression):
    __slots__ = ("type", "n_elements")
    type: Type
    n_elements: Expression


@dataclass(frozen=True)
class ArrayReallocate(Expression):
    __slots__ = ("old", "type", "m_elements")
    old: Assignable
    type: Type
    n_elements: Expression


@dataclass(frozen=True)
class Free(Statement):
    __slots__ = ("target",)
    target: Assignable


@dataclass(frozen=True)
class Declaration(Statement):
    __slots__ = ("name", "type")
    name: Variable
    type: Type

    def assign(self, value: Union[Expression, int, str]):
        value = to_expression(value)
        return DeclarationAssignment(self, value)


@dataclass(frozen=True)
class Assignment(Statement):
    __slots__ = ("target", "value")
    target: Assignable
    value: Expression


@dataclass(frozen=True)
class DeclarationAssignment(Statement):
    __slots__ = ("target", "value")
    target: Declaration
    value: Expression


@dataclass(frozen=True)
class Block(Statement):
    statements: list[Statement]
    comment: Optional[str] = None

    def is_empty(self):
        return len(self.statements) == 0


@dataclass(frozen=True)
class Branch(Statement):
    __slots__ = ("condition", "if_true", "if_false")
    condition: Expression
    if_true: Statement
    if_false: Statement

    @staticmethod
    def join(leaves: List[Tuple[Union[Expression, int, str], Statement]]):
        # This is a fold right operation
        return reduce(
            lambda previous, leaf: Branch(to_expression(leaf[0]), leaf[1], previous),
            reversed(leaves),
            Block([]),
        )


@dataclass(frozen=True)
class Loop(Statement):
    __slots__ = ("condition", "body")
    condition: Expression
    body: Statement


@dataclass(frozen=True)
class Break(Statement):
    __slots__ = ()


@dataclass(frozen=True)
class Return(Statement):
    __slots__ = ("value",)
    value: Expression


@dataclass(frozen=True)
class FunctionDefinition(Statement):
    __slots__ = ("name", "parameters", "return_type", "body")
    name: Variable
    parameters: list[Declaration]
    return_type: Type
    body: Statement


def to_expression(expression: Union[Expression, int, str]) -> Expression:
    if isinstance(expression, int):
        expression = IntegerLiteral(expression)
    elif isinstance(expression, str):
        expression = Variable(expression)
    return expression
