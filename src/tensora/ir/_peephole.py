"""Peephole optimizations.

Current optimization implemented:
* add_zero: 0 + expr or expr + 0 => expr
* minus_zero: expr - 0 => expr
* multiply_zero: 0 * expr or expr * 0 => 0
* multiply_one: 1 * expr or expr * 1 => expr
* equal_same: a == a or a <= a or a >= a => true
* not_equal_same: a != a or a < a or a > a => false
* and_true: expr & true or true & expr => expr
* and_false: expr & false or false & expr => false
* or_true: expr | true or true | expr => true
* or_false: expr | false or false | expr => expr
* boolean_cast_constant: cast(false) => 0 or cast(true) => 1
* branch_true: if (true) then block1 else block2 => block1
* branch_false: if (false) then block1 else block2 => block2
* loop_false: while(false) block => {}
* empty_block: empty blocks get deleted in blocks, branches, and loops
* redundant_assignment: a = a => {}
"""

__all__ = ["peephole_function_definition", "peephole_statement", "peephole"]

from dataclasses import replace
from functools import singledispatch

from .ast import (
    Add,
    And,
    ArrayAllocate,
    ArrayIndex,
    ArrayReallocate,
    Assignable,
    Assignment,
    AttributeAccess,
    Block,
    BooleanLiteral,
    BooleanToInteger,
    Branch,
    Declaration,
    DeclarationAssignment,
    Equal,
    Expression,
    FloatLiteral,
    FunctionDefinition,
    GreaterThan,
    GreaterThanOrEqual,
    IntegerLiteral,
    LessThan,
    LessThanOrEqual,
    Loop,
    Max,
    Min,
    Module,
    Multiply,
    NotEqual,
    Or,
    Return,
    Statement,
    Subtract,
    Variable,
)


@singledispatch
def peephole_assignable(self: Assignable) -> Assignable:
    raise NotImplementedError(f"peephole_assignable not implemented for {type(self)}: {self}")


@peephole_assignable.register(Variable)
def peephole_variable(self: Variable) -> Variable:
    return self


@peephole_assignable.register(AttributeAccess)
def peephole_attribute_access(self: AttributeAccess) -> Assignable:
    return replace(self, target=peephole_assignable(self.target))


@peephole_assignable.register(ArrayIndex)
def peephole_array_index(self: ArrayIndex) -> Assignable:
    return ArrayIndex(peephole_assignable(self.target), peephole_expression(self.index))


@singledispatch
def peephole_expression(self: Expression) -> Expression:
    raise NotImplementedError(f"peephole_expression not implemented for {type(self)}: {self}")


@peephole_expression.register(Assignable)
def peephole_expression_assignable(self: Assignable) -> Assignable:
    # Assignables are expressions in their own right
    return peephole_assignable(self)


@peephole_expression.register(IntegerLiteral)
@peephole_expression.register(FloatLiteral)
@peephole_expression.register(BooleanLiteral)
def peephole_noop(self: Expression) -> Expression:
    return self


@peephole_expression.register(Add)
def peephole_add(self: Add) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == IntegerLiteral(0) or left == FloatLiteral(0.0):
        return right
    elif right == IntegerLiteral(0) or right == FloatLiteral(0.0):
        return left
    else:
        return Add(left, right)


@peephole_expression.register(Subtract)
def peephole_subtract(self: Subtract) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if right == IntegerLiteral(0) or right == FloatLiteral(0.0):
        return left
    else:
        return Subtract(left, right)


@peephole_expression.register(Multiply)
def peephole_multiply(self: Multiply) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == IntegerLiteral(0) or right == IntegerLiteral(0):
        return IntegerLiteral(0)
    elif left == FloatLiteral(0.0) or right == FloatLiteral(0.0):
        return FloatLiteral(0.0)
    elif left == IntegerLiteral(1) or left == FloatLiteral(1.0):
        return right
    elif right == IntegerLiteral(1) or right == FloatLiteral(1.0):
        return left
    else:
        return Multiply(left, right)


@peephole_expression.register(Equal)
@peephole_expression.register(GreaterThanOrEqual)
@peephole_expression.register(LessThanOrEqual)
def peephole_equal(self: Equal) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == right:
        return BooleanLiteral(True)
    else:
        # Use replace so the class is retained
        return replace(self, left=left, right=right)


@peephole_expression.register(NotEqual)
@peephole_expression.register(GreaterThan)
@peephole_expression.register(LessThan)
def peephole_not_equal(self: NotEqual) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == right:
        return BooleanLiteral(False)
    else:
        # Use replace so the class is retained
        return replace(self, left=left, right=right)


@peephole_expression.register(And)
def peephole_and(self: And) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == BooleanLiteral(False) or right == BooleanLiteral(False):
        return BooleanLiteral(False)
    elif left == BooleanLiteral(True):
        return right
    elif right == BooleanLiteral(True):
        return left
    else:
        return And(left, right)


@peephole_expression.register(Or)
def peephole_or(self: Or) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    if left == BooleanLiteral(True) or right == BooleanLiteral(True):
        return BooleanLiteral(True)
    elif left == BooleanLiteral(False):
        return right
    elif right == BooleanLiteral(False):
        return left
    else:
        return Or(left, right)


@peephole_expression.register(Max)
@peephole_expression.register(Min)
def peephole_max_min(self: Max) -> Expression:
    left = peephole_expression(self.left)
    right = peephole_expression(self.right)

    # Use replace so the class is retained
    return replace(self, left=left, right=right)


@peephole_expression.register(BooleanToInteger)
def peephole_boolean_to_integer(self: BooleanToInteger) -> Expression:
    expression = peephole_expression(self.expression)

    if expression == BooleanLiteral(False):
        return IntegerLiteral(0)
    elif expression == BooleanLiteral(True):
        return IntegerLiteral(1)
    else:
        return BooleanToInteger(expression)


@peephole_expression.register(ArrayAllocate)
def peephole_array_allocate(self: ArrayAllocate) -> Expression:
    n_elements = peephole_expression(self.n_elements)
    return replace(self, n_elements=n_elements)


@peephole_expression.register(ArrayReallocate)
def peephole_array_reallocate(self: ArrayReallocate) -> Expression:
    old = peephole_assignable(self.old)
    n_elements = peephole_expression(self.n_elements)
    return replace(self, old=old, n_elements=n_elements)


@singledispatch
def peephole_statement(self: Statement) -> Statement:
    raise NotImplementedError(f"peephole not implemented for {type(self)}: {self}")


@peephole_statement.register(Expression)
def peephole_expression_statement(self: Expression) -> Expression:
    # Expressions are statements in their own right
    return peephole_expression(self)


@peephole_statement.register(Declaration)
def peephole_declaration(self: Declaration) -> Declaration:
    return self


@peephole_statement.register(Assignment)
def peephole_assignment(self: Assignment) -> Statement:
    target = peephole_assignable(self.target)
    value = peephole_expression(self.value)

    if target == value:
        return Block([])
    else:
        return Assignment(target, value)


@peephole_statement.register(DeclarationAssignment)
def peephole_declaration_assignment(self: DeclarationAssignment) -> Statement:
    value = peephole_expression(self.value)
    return replace(self, value=value)


@peephole_statement.register(Block)
def peephole_block(self: Block) -> Statement:
    statements = []
    for old_statement in self.statements:
        statement = peephole_statement(old_statement)
        if isinstance(statement, Block) and statement.is_empty():
            pass
        else:
            statements.append(statement)

    return replace(self, statements=statements)


@peephole_statement.register(Branch)
def peephole_branch(self: Branch) -> Statement:
    condition = peephole_expression(self.condition)
    if_true = peephole_statement(self.if_true)
    if_false = peephole_statement(self.if_false)

    if condition == BooleanLiteral(True):
        return if_true
    elif condition == BooleanLiteral(False):
        return if_false
    elif (
        isinstance(if_true, Block)
        and if_true.is_empty()
        and isinstance(if_false, Block)
        and if_false.is_empty()
    ):
        return Block([])
    else:
        return Branch(condition, if_true, if_false)


@peephole_statement.register(Loop)
def peephole_loop(self: Loop) -> Statement:
    condition = peephole_expression(self.condition)
    body = peephole_statement(self.body)

    if condition == BooleanLiteral(False):
        return Block([])
    elif isinstance(self.body, Block) and self.body.is_empty():
        return Block([])
    else:
        return Loop(condition, body)


@peephole_statement.register(Return)
def peephole_return(self: Return) -> Statement:
    value = peephole_expression(self.value)
    return Return(value)


def peephole_function_definition(self: FunctionDefinition) -> FunctionDefinition:
    body = peephole_statement(self.body)
    return replace(self, body=body)


def peephole(self: Module) -> Module:
    functions = [peephole_function_definition(function) for function in self.definitions]
    return Module(functions)
