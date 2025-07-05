__all__ = ["hoist_declarations"]

from functools import singledispatch

from ..ir.ast import (
    Assignment,
    Block,
    Branch,
    Declaration,
    DeclarationAssignment,
    Expression,
    FunctionDefinition,
    Loop,
    Return,
    Statement,
)
from ..ir.types import Type


@singledispatch
def hoist_declarations_statement(self: Statement) -> dict[str, Type]:
    raise NotImplementedError(
        f"hoist_declarations_statement not implemented for {type(self).__name__}"
    )


@hoist_declarations_statement.register
def hoist_declarations_expression(self: Expression) -> dict[str, Type]:
    return {}


@hoist_declarations_statement.register
def hoist_declarations_declaration(self: Declaration) -> dict[str, Type]:
    return {self.name.name: self.type}


@hoist_declarations_statement.register
def hoist_declarations_assignment(self: Assignment) -> dict[str, Type]:
    return {}


@hoist_declarations_statement.register
def hoist_declarations_declaration_assignment(self: DeclarationAssignment) -> dict[str, Type]:
    return {self.target.name.name: self.target.type}


@hoist_declarations_statement.register
def hoist_declarations_block(self: Block) -> dict[str, Type]:
    result = {}
    for s in self.statements:
        result.update(hoist_declarations_statement(s))
    return result


@hoist_declarations_statement.register
def hoist_declarations_branch(self: Branch) -> dict[str, Type]:
    result = {}
    result.update(hoist_declarations_statement(self.if_true))
    result.update(hoist_declarations_statement(self.if_false))
    return result


@hoist_declarations_statement.register
def hoist_declarations_loop(self: Loop) -> dict[str, Type]:
    return hoist_declarations_statement(self.body)


@hoist_declarations_statement.register
def hoist_declarations_return(self: Return) -> dict[str, Type]:
    return {}


def hoist_declarations(fn: FunctionDefinition) -> dict[str, Type]:
    return hoist_declarations_statement(fn.body)
