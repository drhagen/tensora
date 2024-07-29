__all__ = ["ir_to_c_statement", "ir_to_c_function_definition", "ir_to_c"]

from functools import singledispatch

from ..ir.ast import (
    Add,
    And,
    ArrayAllocate,
    ArrayIndex,
    ArrayReallocate,
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
from ._type_to_c import type_to_c


def parens(code: Expression, wrap_me: type | tuple[type, ...]):
    string = ir_to_c_expression(code)
    if isinstance(code, wrap_me):
        string = f"({string})"
    return string


def indent_lines(lines: list[str]) -> list[str]:
    return ["  " + line for line in lines]


@singledispatch
def ir_to_c_expression(self: Expression) -> str:
    raise NotImplementedError(f"ir_to_c_expression not implemented for {type(self)}: {self}")


@ir_to_c_expression.register(Variable)
def ir_to_c_variable(self: Variable) -> str:
    return self.name


@ir_to_c_expression.register(AttributeAccess)
def ir_to_c_attribute_access(self: AttributeAccess) -> str:
    return f"{ir_to_c_expression(self.target)}->{self.attribute}"


@ir_to_c_expression.register(ArrayIndex)
def ir_to_c_array_index(self: ArrayIndex) -> str:
    return f"{ir_to_c_expression(self.target)}[{ir_to_c_expression(self.index)}]"


@ir_to_c_expression.register(IntegerLiteral)
def ir_to_c_integer_literal(self: IntegerLiteral) -> str:
    return str(self.value)


@ir_to_c_expression.register(FloatLiteral)
def ir_to_c_float_literal(self: FloatLiteral) -> str:
    return str(self.value)


@ir_to_c_expression.register(BooleanLiteral)
def ir_to_c_boolean_literal(self: BooleanLiteral) -> str:
    return str(int(self.value))


@ir_to_c_expression.register(Add)
def ir_to_c_add(self: Add) -> str:
    return f"{ir_to_c_expression(self.left)} + {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(Subtract)
def ir_to_c_subtract(self: Subtract) -> str:
    # Subtract does not have the associative property so it needs parentheses around the right operand if it has the
    # same precedence.
    return f"{ir_to_c_expression(self.left)} - {parens(self.right, (Add, Subtract))}"


@ir_to_c_expression.register(Multiply)
def ir_to_c_multiply(self: Multiply) -> str:
    return f"{parens(self.left, (Add, Subtract))} * {parens(self.right, (Add, Subtract))}"


@ir_to_c_expression.register(Equal)
def ir_to_c_equal(self: Equal) -> str:
    return f"{ir_to_c_expression(self.left)} == {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(NotEqual)
def ir_to_c_not_equal(self: NotEqual) -> str:
    return f"{ir_to_c_expression(self.left)} != {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(GreaterThan)
def ir_to_c_greater_than(self: GreaterThan) -> str:
    return f"{ir_to_c_expression(self.left)} > {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(LessThan)
def ir_to_c_less_than(self: LessThan) -> str:
    return f"{ir_to_c_expression(self.left)} < {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(GreaterThanOrEqual)
def ir_to_c_greater_than_or_equal(self: GreaterThanOrEqual) -> str:
    return f"{ir_to_c_expression(self.left)} >= {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(LessThanOrEqual)
def ir_to_c_less_than_or_equal(self: LessThanOrEqual) -> str:
    return f"{ir_to_c_expression(self.left)} <= {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(And)
def ir_to_c_and(self: And) -> str:
    return f"{parens(self.left, Or)} && {parens(self.right, Or)}"


@ir_to_c_expression.register(Or)
def ir_to_c_or(self: Or) -> str:
    return f"{ir_to_c_expression(self.left)} || {ir_to_c_expression(self.right)}"


@ir_to_c_expression.register(Max)
def ir_to_c_max(self: Max) -> str:
    return f"TACO_MAX({ir_to_c_expression(self.left)}, {ir_to_c_expression(self.right)})"


@ir_to_c_expression.register(Min)
def ir_to_c_min(self: Min) -> str:
    return f"TACO_MIN({ir_to_c_expression(self.left)}, {ir_to_c_expression(self.right)})"


@ir_to_c_expression.register(BooleanToInteger)
def ir_to_c_boolean_to_integer(self: BooleanToInteger) -> str:
    return f"(int32_t)({ir_to_c_expression(self.expression)})"


@ir_to_c_expression.register(ArrayAllocate)
def ir_to_c_array_allocate(self: ArrayAllocate) -> str:
    return f"malloc(sizeof({type_to_c(self.element_type)}) * {parens(self.n_elements, (Add, Subtract))})"


@ir_to_c_expression.register(ArrayReallocate)
def ir_to_c_array_reallocate(self: ArrayReallocate) -> str:
    old = ir_to_c_expression(self.old)
    return f"realloc({old}, sizeof({type_to_c(self.element_type)}) * {parens(self.n_elements, (Add, Subtract))})"


def ir_to_c_declaration(self: Declaration) -> str:
    return type_to_c(self.type, self.name.name)


@singledispatch
def ir_to_c_statement(self: Statement) -> list[str]:
    raise NotImplementedError(f"ir_to_c_statement not implemented for {type(self)}: {self}")


@ir_to_c_statement.register(Expression)
def convert_expression_to_statement(self: Expression) -> list[str]:
    # Every expression can also be a statement; convert it here
    return [ir_to_c_expression(self) + ";"]


@ir_to_c_statement.register(Declaration)
def convert_declaration_to_statement(self: Declaration) -> list[str]:
    return [ir_to_c_declaration(self) + ";"]


@ir_to_c_statement.register(Assignment)
def ir_to_c_assignment(self: Assignment) -> list[str]:
    target = ir_to_c_expression(self.target)
    if isinstance(self.value, Add) and self.value.left == self.target:
        if self.value.right == IntegerLiteral(1):
            return [f"{target}++;"]
        else:
            return [f"{target} += {ir_to_c_expression(self.value.right)};"]
    elif isinstance(self.value, Subtract) and self.value.left == self.target:
        if self.value.right == IntegerLiteral(1):
            return [f"{target}--;"]
        else:
            return [f"{target} -= {ir_to_c_expression(self.value.right)};"]
    elif isinstance(self.value, Multiply) and self.value.left == self.target:
        return [f"{target} *= {ir_to_c_expression(self.value.right)};"]
    else:
        return [f"{target} = {ir_to_c_expression(self.value)};"]


@ir_to_c_statement.register(DeclarationAssignment)
def ir_to_c_declaration_assignment(self: DeclarationAssignment) -> list[str]:
    return [f"{ir_to_c_declaration(self.target)} = {ir_to_c_expression(self.value)};"]


@ir_to_c_statement.register(Block)
def ir_to_c_block(self: Block) -> list[str]:
    lines = []
    need_separator = False

    if self.comment is not None:
        # Add comment if it there is one
        lines.append(f"// {self.comment}")

    for statement in self.statements:
        if isinstance(statement, Block):
            if len(lines) > 0:
                # Add blank line if there are preceding lines
                lines.append("")
            lines.extend(ir_to_c_block(statement))
            need_separator = True
        else:
            if need_separator:
                # A block immediately preceded this line, so add the separator
                lines.append("")
                need_separator = False
            lines.extend(ir_to_c_statement(statement))

    return lines


@ir_to_c_statement.register(Branch)
def ir_to_c_branch(self: Branch) -> list[str]:
    if_true_lines = ir_to_c_statement(self.if_true)
    if_false_lines = ir_to_c_statement(self.if_false)

    lines = []
    lines.append(f"if ({ir_to_c_expression(self.condition)}) {{")
    lines.extend(indent_lines(if_true_lines))

    if isinstance(self.if_false, Branch):
        # Special case if-else chain to be put at the same level
        lines.append(f"}} else {if_false_lines[0]}")
        lines.extend(if_false_lines[1:])
    elif self.if_false == Block([]):
        # Special case empty if_false block to emit no else branch
        lines.append("}")
    else:
        lines.append("} else {")
        lines.extend(indent_lines(if_false_lines))
        lines.append("}")

    return lines


@ir_to_c_statement.register(Loop)
def ir_to_c_loop(self: Loop) -> list[str]:
    return [
        f"while ({ir_to_c_expression(self.condition)}) {{",
        *indent_lines(ir_to_c_statement(self.body)),
        "}",
    ]


@ir_to_c_statement.register(Return)
def ir_to_c_return(self: Return) -> list[str]:
    return [f"return {ir_to_c_expression(self.value)};"]


def ir_to_c_function_definition(self: FunctionDefinition) -> str:
    return_type_string = type_to_c(self.return_type)
    name_string = ir_to_c_expression(self.name)
    parameters_string = ", ".join(map(ir_to_c_declaration, self.parameters))

    lines = [
        f"{return_type_string} {name_string}({parameters_string}) {{",
        *indent_lines(ir_to_c_statement(self.body)),
        "}",
    ]

    return "\n".join(lines)


def ir_to_c(self: Module) -> str:
    return "\n\n".join((ir_to_c_function_definition(function) for function in self.definitions))
