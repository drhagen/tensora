__all__ = ["ast_to_c"]

from functools import singledispatch
from typing import Union

from ..ir.ast import *
from .type_to_c import type_to_c


def parens(code: Expression, wrap_me: Union[type, tuple[type, ...]]):
    string = ast_to_c_expression(code)
    if isinstance(code, wrap_me):
        string = f"({string})"
    return string


def indent_lines(lines: list[str]) -> list[str]:
    return ["  " + line for line in lines]


def ast_to_c(code: Statement) -> str:
    return "\n".join(ast_to_c_statement(code))


@singledispatch
def ast_to_c_expression(code: Expression) -> str:
    raise NotImplementedError(f"No implementation of ast_to_c_expression: {code}")


@ast_to_c_expression.register(Variable)
def ast_to_c_variable(code: Variable):
    return code.name


@ast_to_c_expression.register(AttributeAccess)
def ast_to_c_attribute_access(code: AttributeAccess):
    return f"{ast_to_c_expression(code.target)}->{code.attribute}"


@ast_to_c_expression.register(ArrayIndex)
def ast_to_c_array_index(code: ArrayIndex):
    return f"{ast_to_c_expression(code.target)}[{ast_to_c_expression(code.index)}]"


@ast_to_c_expression.register(IntegerLiteral)
def ast_to_c_integer_literal(code: IntegerLiteral):
    return str(code.value)


@ast_to_c_expression.register(FloatLiteral)
def ast_to_c_float_literal(code: FloatLiteral):
    return str(code.value)


@ast_to_c_expression.register(BooleanLiteral)
def ast_to_c_boolean_literal(code: BooleanLiteral):
    return str(int(code.value))


@ast_to_c_expression.register(ArrayLiteral)
def ast_to_c_array_literal(code: ArrayLiteral):
    return "{" + ", ".join(map(ast_to_c_expression, code.elements)) + "}"


@ast_to_c_expression.register(Add)
def ast_to_c_add(code: Add):
    return f"{ast_to_c_expression(code.left)} + {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(Subtract)
def ast_to_c_subtract(code: Subtract):
    # Subtract does not have the associative property so it needs parentheses around the right operand if it has the
    # same precedence.
    return f"{ast_to_c_expression(code.left)} - {parens(code.right, (Add, Subtract))}"


@ast_to_c_expression.register(Multiply)
def ast_to_c_multiply(code: Multiply):
    return f"{parens(code.left, (Add, Subtract))} * {parens(code.right, (Add, Subtract))}"


@ast_to_c_expression.register(Equal)
def ast_to_c_equal(code: Equal):
    return f"{ast_to_c_expression(code.left)} == {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(NotEqual)
def ast_to_c_not_equal(code: NotEqual):
    return f"{ast_to_c_expression(code.left)} != {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(GreaterThan)
def ast_to_c_greater_than(code: GreaterThan):
    return f"{ast_to_c_expression(code.left)} > {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(LessThan)
def ast_to_c_less_than(code: LessThan):
    return f"{ast_to_c_expression(code.left)} < {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(GreaterThanOrEqual)
def ast_to_c_greater_than_or_equal(code: GreaterThanOrEqual):
    return f"{ast_to_c_expression(code.left)} >= {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(LessThanOrEqual)
def ast_to_c_less_than_or_equal(code: LessThanOrEqual):
    return f"{ast_to_c_expression(code.left)} <= {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(And)
def ast_to_c_and(code: And):
    return f"{parens(code.left, Or)} && {parens(code.right, Or)}"


@ast_to_c_expression.register(Or)
def ast_to_c_or(code: Or):
    return f"{ast_to_c_expression(code.left)} || {ast_to_c_expression(code.right)}"


@ast_to_c_expression.register(FunctionCall)
def ast_to_c_function_call(code: FunctionCall):
    return f"{ast_to_c_expression(code.name)}({', '.join(map(ast_to_c_expression, code.arguments))})"


@ast_to_c_expression.register(Max)
def ast_to_c_max(code: Max):
    return f"TACO_MAX({ast_to_c_expression(code.left)}, {ast_to_c_expression(code.right)})"


@ast_to_c_expression.register(Min)
def ast_to_c_min(code: Min):
    return f"TACO_MIN({ast_to_c_expression(code.left)}, {ast_to_c_expression(code.right)})"


@ast_to_c_expression.register(BooleanToInteger)
def ast_to_c_boolean_to_integer(code: BooleanToInteger):
    return f"(int32_t)({ast_to_c_expression(code.expression)})"


@ast_to_c_expression.register(Allocate)
def ast_to_c_allocate(code: Allocate):
    return f"malloc(sizeof({type_to_c(code.type)}))"


@ast_to_c_expression.register(ArrayAllocate)
def ast_to_c_array_allocate(code: ArrayAllocate):
    return f"malloc(sizeof({type_to_c(code.type)}) * {parens(code.n_elements, (Add, Subtract))})"


@ast_to_c_expression.register(ArrayReallocate)
def ast_to_c_array_reallocate(code: ArrayReallocate):
    old = ast_to_c_expression(code.old)
    return f"realloc({old}, sizeof({type_to_c(code.type)}) * {parens(code.n_elements, (Add, Subtract))})"


def ast_to_c_declaration(code: Declaration):
    return type_to_c(code.type, code.name.name)


@singledispatch
def ast_to_c_statement(code: Statement) -> list[str]:
    raise NotImplementedError(f"No implementation of ast_to_c_statement: {code}")


@ast_to_c_statement.register(Expression)
def convert_expression_to_statement(code: Expression):
    # Every expression can also be a statement; convert it here
    return [ast_to_c_expression(code) + ";"]


@ast_to_c_statement.register(Declaration)
def convert_declaration_to_statement(code: Declaration):
    return [ast_to_c_declaration(code) + ";"]


@ast_to_c_statement.register(Assignment)
def ast_to_c_assignment(code: Assignment):
    target = ast_to_c_expression(code.target)
    if isinstance(code.value, Add) and code.value.left == code.target:
        if code.value.right == IntegerLiteral(1):
            return [f"{target}++;"]
        else:
            return [f"{target} += {ast_to_c_expression(code.value.right)};"]
    elif isinstance(code.value, Subtract) and code.value.left == code.target:
        if code.value.right == IntegerLiteral(1):
            return [f"{target}--;"]
        else:
            return [f"{target} -= {ast_to_c_expression(code.value.right)};"]
    elif isinstance(code.value, Multiply) and code.value.left == code.target:
        return [f"{target} *= {ast_to_c_expression(code.value.right)};"]
    else:
        return [f"{target} = {ast_to_c_expression(code.value)};"]


@ast_to_c_statement.register(DeclarationAssignment)
def ast_to_c_assignment(code: DeclarationAssignment):
    return [f"{ast_to_c_declaration(code.target)} = {ast_to_c_expression(code.value)};"]


@ast_to_c_statement.register(Block)
def ast_to_c_block(code: Block):
    lines = []
    need_separator = False

    if code.comment is not None:
        # Add comment if it there is one
        lines.append(f"// {code.comment}")

    for i, statement in enumerate(code.statements):
        if isinstance(statement, Block):
            if len(lines) > 0:
                # Add blank line if there are preceding lines
                lines.append("")
            lines.extend(ast_to_c_block(statement))
            need_separator = True
        else:
            if need_separator:
                # A block immediately preceded this line, so add the separator
                lines.append("")
                need_separator = False
            lines.extend(ast_to_c_statement(statement))

    return lines


@ast_to_c_statement.register(Branch)
def ast_to_c_branch(code: Branch):
    if_true_lines = ast_to_c_statement(code.if_true)
    if_false_lines = ast_to_c_statement(code.if_false)

    lines = []
    lines.append(f"if ({ast_to_c_expression(code.condition)}) {{")
    lines.extend(indent_lines(if_true_lines))

    if isinstance(code.if_false, Branch):
        # Special case if-else chain to be put at the same level
        lines.append(f"}} else {if_false_lines[0]}")
        lines.extend(if_false_lines[1:])
    elif code.if_false == Block([]):
        # Special case empty if_false block to emit no else branch
        lines.append("}")
    else:
        lines.append("} else {")
        lines.extend(indent_lines(if_false_lines))
        lines.append("}")

    return lines


@ast_to_c_statement.register(Loop)
def ast_to_c_loop(code: Loop):
    return [
        f"while ({ast_to_c_expression(code.condition)}) {{",
        *indent_lines(ast_to_c_statement(code.body)),
        "}",
    ]


@ast_to_c_statement.register(Return)
def ast_to_c_return(code: Return):
    return [f"return {ast_to_c_expression(code.value)};"]


@ast_to_c_statement.register(FunctionDefinition)
def ast_to_c_function_definition(code: FunctionDefinition):
    return_type_string = type_to_c(code.return_type)
    name_string = ast_to_c_expression(code.name)
    parameters_string = ", ".join(map(ast_to_c_declaration, code.parameters))
    return [
        f"{return_type_string} {name_string}({parameters_string}) {{",
        *indent_lines(ast_to_c_statement(code.body)),
        "}",
    ]
