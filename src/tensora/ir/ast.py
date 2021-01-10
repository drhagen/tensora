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
    "BooleanToInteger",
    "Allocate",
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
]

from dataclasses import dataclass
from typing import Optional

from .types import Type


class Statement:
    pass


class Expression(Statement):
    pass


class Assignable(Expression):
    pass


@dataclass(frozen=True)
class Variable(Assignable):
    __slots__ = ("name",)
    name: str


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
class ArrayLiteral(Expression):
    __slots__ = ("value",)
    elements: list[Expression]


@dataclass(frozen=True)
class Add(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


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


@dataclass(frozen=True)
class Or(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


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


@dataclass(frozen=True)
class Min(Expression):
    __slots__ = ("left", "right")
    left: Expression
    right: Expression


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
class Declaration(Statement):
    __slots__ = ("name", "type")
    name: Variable
    type: Type


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


@dataclass(frozen=True)
class Loop(Statement):
    __slots__ = ("condition", "body")
    condition: Expression
    body: Statement


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
