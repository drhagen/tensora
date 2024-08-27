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
    "Max",
    "Min",
    "BooleanToInteger",
    "ArrayAllocate",
    "ArrayReallocate",
    "Declaration",
    "Assignment",
    "DeclarationAssignment",
    "Block",
    "Branch",
    "Loop",
    "Return",
    "FunctionDefinition",
    "Module",
]

from dataclasses import dataclass
from functools import reduce
from typing import Sequence

from .types import Type


def to_expression(expression: Expression | float | int | str) -> Expression:
    match expression:
        case Expression():
            return expression
        case float():
            return FloatLiteral(expression)
        case int():
            return IntegerLiteral(expression)
        case str():
            return Variable(expression)


class Statement:
    __slots__ = ()


class Expression(Statement):
    __slots__ = ()

    def plus(self, term: Expression | int | str) -> Expression:
        term = to_expression(term)
        return Add(self, term)

    def minus(self, term: Expression | int | str) -> Expression:
        term = to_expression(term)
        return Subtract(self, term)

    def times(self, factor: Expression | int | str) -> Expression:
        factor = to_expression(factor)
        return Multiply(self, factor)


class Assignable(Expression):
    __slots__ = ()

    def attr(self, attribute: str) -> Assignable:
        return AttributeAccess(self, attribute)

    def idx(self, index: Expression | int | str) -> Assignable:
        index = to_expression(index)
        return ArrayIndex(self, index)

    def assign(self, value: Expression | float | int | str) -> Statement:
        value = to_expression(value)
        return Assignment(self, value)

    def increment(self, amount: Expression | float | int | str = 1) -> Statement:
        amount = to_expression(amount)
        return self.assign(self.plus(amount))


@dataclass(frozen=True, slots=True)
class Variable(Assignable):
    name: str

    def declare(self, type: Type) -> Declaration:
        return Declaration(self, type)


@dataclass(frozen=True, slots=True)
class AttributeAccess(Assignable):
    # For languages that care, this represents attribute access to an object on
    # the heap. There is no way to use an object on the stack in this IR.
    target: Assignable
    attribute: str


@dataclass(frozen=True, slots=True)
class ArrayIndex(Assignable):
    # For languages that care, this represents attribute access to an array on
    # the heap. There is no way to use an array on the stack in this IR.
    target: Assignable
    index: Expression


@dataclass(frozen=True, slots=True)
class IntegerLiteral(Expression):
    value: int


@dataclass(frozen=True, slots=True)
class FloatLiteral(Expression):
    value: float


@dataclass(frozen=True, slots=True)
class BooleanLiteral(Expression):
    value: bool


@dataclass(frozen=True, slots=True)
class Add(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(Add, expression_operands, IntegerLiteral(0))


@dataclass(frozen=True, slots=True)
class Subtract(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(Multiply, expression_operands, IntegerLiteral(1))


@dataclass(frozen=True, slots=True)
class Equal(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class NotEqual(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class GreaterThan(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class LessThan(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class GreaterThanOrEqual(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class LessThanOrEqual(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class And(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(And, expression_operands, BooleanLiteral(True))


@dataclass(frozen=True, slots=True)
class Or(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(Or, expression_operands, BooleanLiteral(False))


@dataclass(frozen=True, slots=True)
class Max(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(Max, expression_operands)


@dataclass(frozen=True, slots=True)
class Min(Expression):
    left: Expression
    right: Expression

    @staticmethod
    def join(operands: Sequence[Expression | int | str]) -> Expression:
        expression_operands = [to_expression(operand) for operand in operands]
        return reduce(Min, expression_operands)


@dataclass(frozen=True, slots=True)
class BooleanToInteger(Expression):
    expression: Expression


@dataclass(frozen=True, slots=True)
class ArrayAllocate(Expression):
    element_type: Type
    n_elements: Expression


@dataclass(frozen=True, slots=True)
class ArrayReallocate(Expression):
    old: Assignable
    element_type: Type
    n_elements: Expression


@dataclass(frozen=True, slots=True)
class Declaration(Statement):
    name: Variable
    type: Type

    def assign(self, value: Expression | int | str) -> Statement:
        value = to_expression(value)
        return DeclarationAssignment(self, value)


@dataclass(frozen=True, slots=True)
class Assignment(Statement):
    target: Assignable
    value: Expression


@dataclass(frozen=True, slots=True)
class DeclarationAssignment(Statement):
    target: Declaration
    value: Expression


@dataclass(frozen=True, slots=True)
class Block(Statement):
    statements: list[Statement]
    comment: str | None = None

    def is_empty(self):
        return len(self.statements) == 0


@dataclass(frozen=True, slots=True)
class Branch(Statement):
    condition: Expression
    if_true: Statement
    if_false: Statement

    @staticmethod
    def join(leaves: Sequence[tuple[Expression | int | str, Statement]]) -> Statement:
        # This is a fold right operation
        return reduce(
            lambda previous, leaf: Branch(to_expression(leaf[0]), leaf[1], previous),
            reversed(leaves),
            Block([]),
        )


@dataclass(frozen=True, slots=True)
class Loop(Statement):
    condition: Expression
    body: Statement


@dataclass(frozen=True, slots=True)
class Return(Statement):
    value: Expression


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    name: Variable
    parameters: list[Declaration]
    return_type: Type
    body: Statement


@dataclass(frozen=True, slots=True)
class Module:
    definitions: list[FunctionDefinition]
