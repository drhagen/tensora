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
"""

__all__ = ["peephole"]

from dataclasses import replace
from functools import singledispatch

from .ast import (
    Add,
    Address,
    Allocate,
    And,
    ArrayAllocate,
    ArrayIndex,
    ArrayLiteral,
    ArrayReallocate,
    Assignable,
    Assignment,
    AttributeAccess,
    Block,
    BooleanLiteral,
    BooleanToInteger,
    Branch,
    Break,
    Declaration,
    DeclarationAssignment,
    Equal,
    Expression,
    FloatLiteral,
    Free,
    FunctionCall,
    FunctionDefinition,
    GreaterThan,
    GreaterThanOrEqual,
    IntegerLiteral,
    LessThan,
    LessThanOrEqual,
    Loop,
    Max,
    Min,
    ModeLiteral,
    Multiply,
    NotEqual,
    Or,
    Return,
    Statement,
    Subtract,
    Variable,
)


@singledispatch
def peephole(code: Statement) -> Statement:
    raise NotImplementedError(f"No implementation of peephole_statement: {code}")


@peephole.register(Expression)
@singledispatch
def peephole_expression(code: Expression) -> Expression:
    raise NotImplementedError(f"No implementation of peephole_expression: {code}")


@peephole_expression.register(Assignable)
@singledispatch
def peephole_assignable(code: Assignable) -> Assignable:
    raise NotImplementedError(f"No implementation of peephole_assignable: {code}")


@peephole_assignable.register(Variable)
def peephole_variable(code: Variable):
    return code


@peephole_assignable.register(AttributeAccess)
def peephole_attribute_access(code: AttributeAccess):
    return replace(code, target=peephole_assignable(code.target))


@peephole_assignable.register(ArrayIndex)
def peephole_array_index(code: ArrayIndex):
    return ArrayIndex(peephole_assignable(code.target), peephole_expression(code.index))


@peephole_expression.register(IntegerLiteral)
@peephole_expression.register(FloatLiteral)
@peephole_expression.register(BooleanLiteral)
@peephole_expression.register(ModeLiteral)
@peephole_expression.register(Address)
@peephole_expression.register(Allocate)
def peephole_noop(code: Expression):
    return code


@peephole_expression.register(ArrayLiteral)
def peephole_array_literal(code: ArrayLiteral):
    return ArrayLiteral([peephole_expression(element) for element in code.elements])


@peephole_expression.register(Add)
def peephole_add(code: Add):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if left == IntegerLiteral(0) or left == FloatLiteral(0.0):
        return right
    elif right == IntegerLiteral(0) or right == FloatLiteral(0.0):
        return left
    else:
        return Add(left, right)


@peephole_expression.register(Subtract)
def peephole_subtract(code: Subtract):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if right == IntegerLiteral(0) or right == FloatLiteral(0.0):
        return left
    else:
        return Subtract(left, right)


@peephole_expression.register(Multiply)
def peephole_multiply(code: Multiply):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

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
def peephole_equal(code: Equal):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if left == right:
        return BooleanLiteral(True)
    else:
        # Use replace so the class is retained
        return replace(code, left=left, right=right)


@peephole_expression.register(NotEqual)
@peephole_expression.register(GreaterThan)
@peephole_expression.register(LessThan)
def peephole_not_equal(code: NotEqual):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if left == right:
        return BooleanLiteral(False)
    else:
        # Use replace so the class is retained
        return replace(code, left=left, right=right)


@peephole_expression.register(And)
def peephole_and(code: And):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if left == BooleanLiteral(False) or right == BooleanLiteral(False):
        return BooleanLiteral(False)
    elif left == BooleanLiteral(True):
        return right
    elif right == BooleanLiteral(True):
        return left
    else:
        return And(left, right)


@peephole_expression.register(Or)
def peephole_or(code: Or):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    if left == BooleanLiteral(True) or right == BooleanLiteral(True):
        return BooleanLiteral(True)
    elif left == BooleanLiteral(False):
        return right
    elif right == BooleanLiteral(False):
        return left
    else:
        return Or(left, right)


@peephole_expression.register(FunctionCall)
def peephole_function_call(code: FunctionCall):
    arguments = [peephole_expression(argument) for argument in code.arguments]
    return replace(code, arguments=arguments)


@peephole_expression.register(Max)
@peephole_expression.register(Min)
def peephole_max_min(code: Max):
    left = peephole_expression(code.left)
    right = peephole_expression(code.right)

    # Use replace so the class is retained
    return replace(code, left=left, right=right)


@peephole_expression.register(BooleanToInteger)
def peephole_boolean_to_integer(code: BooleanToInteger):
    expression = peephole_expression(code.expression)

    if expression == BooleanLiteral(False):
        return IntegerLiteral(0)
    elif expression == BooleanLiteral(True):
        return IntegerLiteral(1)
    else:
        return BooleanToInteger(expression)


@peephole_expression.register(ArrayAllocate)
def peephole_array_allocate(code: ArrayAllocate):
    n_elements = peephole_expression(code.n_elements)
    return replace(code, n_elements=n_elements)


@peephole_expression.register(ArrayReallocate)
def peephole_array_reallocate(code: ArrayReallocate):
    old = peephole_assignable(code.old)
    n_elements = peephole_expression(code.n_elements)
    return replace(code, old=old, n_elements=n_elements)


@peephole.register(Declaration)
def peephole_declaration(code: Declaration):
    return code


@peephole.register(Free)
def peephole_free(code: Free):
    return Free(peephole_assignable(code.target))


@peephole.register(Assignment)
def peephole_assignment(code: Assignment):
    target = peephole_assignable(code.target)
    value = peephole_expression(code.value)
    return Assignment(target, value)


@peephole.register(DeclarationAssignment)
def peephole_declaration_assignment(code: DeclarationAssignment):
    value = peephole_expression(code.value)
    return replace(code, value=value)


@peephole.register(Block)
def peephole_block(code: Block):
    statements = []
    for old_statement in code.statements:
        statement = peephole(old_statement)
        if isinstance(statement, Block) and statement.is_empty():
            pass
        else:
            statements.append(statement)

    return replace(code, statements=statements)


@peephole.register(Branch)
def peephole_branch(code: Branch):
    condition = peephole_expression(code.condition)
    if_true = peephole(code.if_true)
    if_false = peephole(code.if_false)

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


@peephole.register(Loop)
def peephole_loop(code: Loop):
    condition = peephole_expression(code.condition)
    body = peephole(code.body)

    if condition == BooleanLiteral(False):
        return Block([])
    elif isinstance(code.body, Block) and code.body.is_empty():
        return Block([])
    else:
        return Loop(condition, body)


@peephole.register(Break)
def peephole_break(code: Break):
    return code


@peephole.register(Return)
def peephole_return(code: Return):
    value = peephole_expression(code.value)
    return Return(value)


@peephole.register(FunctionDefinition)
def peephole_function_definition(code: FunctionDefinition):
    body = peephole(code.body)
    return replace(code, body=body)
