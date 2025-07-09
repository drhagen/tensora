__all__ = ["ir_to_llvm"]

from functools import singledispatch

import llvmlite.ir as llvm

from ..compile import target
from ..ir.ast import (
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
from ._hoist_declarations import hoist_declarations
from ._type_to_llvm import (
    attribute_indexes,
    llvm_boolean_type,
    llvm_float_type,
    llvm_integer_type,
    llvm_size_type,
    type_to_llvm,
)

# Create a target machine from the target so that we can get the word sizes
target_machine = target.create_target_machine()


@singledispatch
def get_element_pointer(
    self: Assignable, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    raise NotImplementedError(f"get_element_pointer not implemented for {type(self)}: {self}")


@get_element_pointer.register(Variable)
def get_element_pointer_variable(
    self: Variable, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return locals[self.name]


@get_element_pointer.register(AttributeAccess)
def get_element_pointer_attribute_access(
    self: AttributeAccess, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    stem = ir_to_llvm_expression(self.target, builder, locals)
    return builder.gep(
        stem,
        [
            llvm.Constant(llvm_size_type, 0),
            # All structs are indexed with i32 in LLVM
            llvm.Constant(llvm.IntType(32), attribute_indexes[self.attribute]),
        ],
    )


@get_element_pointer.register(ArrayIndex)
def get_element_pointer_array_index(
    self: ArrayIndex, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    target = ir_to_llvm_expression(self.target, builder, locals)
    index = ir_to_llvm_expression(self.index, builder, locals)
    return builder.gep(target, [index])


@singledispatch
def ir_to_llvm_expression(
    self: Expression, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    raise NotImplementedError(f"ir_to_llvm_expression not implemented for {type(self)}: {self}")


@ir_to_llvm_expression.register(Variable)
def ir_to_llvm_variable(
    self: Variable, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.load(locals[self.name])


@ir_to_llvm_expression.register(AttributeAccess)
def ir_to_llvm_attribute_access(
    self: AttributeAccess, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    stem = ir_to_llvm_expression(self.target, builder, locals)
    return builder.load(
        builder.gep(
            stem,
            [
                llvm.Constant(llvm_size_type, 0),
                # All structs are indexed with i32 in LLVM
                llvm.Constant(llvm.IntType(32), attribute_indexes[self.attribute]),
            ],
        )
    )


@ir_to_llvm_expression.register(ArrayIndex)
def ir_to_llvm_array_index(
    self: ArrayIndex, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    target = ir_to_llvm_expression(self.target, builder, locals)
    index = ir_to_llvm_expression(self.index, builder, locals)
    return builder.load(builder.gep(target, [index]))


@ir_to_llvm_expression.register(IntegerLiteral)
def ir_to_llvm_integer_literal(
    self: IntegerLiteral, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return llvm.Constant(llvm_integer_type, self.value)


@ir_to_llvm_expression.register(FloatLiteral)
def ir_to_llvm_float_literal(
    self: FloatLiteral, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return llvm.Constant(llvm_float_type, self.value)


@ir_to_llvm_expression.register(BooleanLiteral)
def ir_to_llvm_boolean_literal(
    self: BooleanLiteral, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return llvm.Constant(llvm_boolean_type, self.value)


@ir_to_llvm_expression.register(Add)
def ir_to_llvm_add(
    self: Add, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left = ir_to_llvm_expression(self.left, builder, locals)
    right = ir_to_llvm_expression(self.right, builder, locals)
    match left.type, right.type:
        case (llvm.IntType(), llvm.IntType()):
            return builder.add(left, right)
        case (llvm.IntType(), llvm.DoubleType()):
            left = builder.sitofp(left, llvm_float_type)
            return builder.fadd(left, right)
        case (llvm.DoubleType(), llvm.IntType()):
            right = builder.sitofp(right, llvm_float_type)
            return builder.fadd(left, right)
        case (llvm.DoubleType(), llvm.DoubleType()):
            return builder.fadd(left, right)
        case (llvm.PointerType(), llvm.IntType()):
            return builder.gep(left, [right])
        case _:
            raise TypeError(f"Cannot add {left.type} and {right.type}")


@ir_to_llvm_expression.register(Subtract)
def ir_to_llvm_subtract(
    self: Subtract, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left = ir_to_llvm_expression(self.left, builder, locals)
    right = ir_to_llvm_expression(self.right, builder, locals)
    match left.type, right.type:
        case (llvm.IntType(), llvm.IntType()):
            return builder.sub(left, right)
        case (llvm.IntType(), llvm.DoubleType()):
            left = builder.sitofp(left, llvm_float_type)
            return builder.fsub(left, right)
        case (llvm.DoubleType(), llvm.IntType()):
            right = builder.sitofp(right, llvm_float_type)
            return builder.fsub(left, right)
        case (llvm.DoubleType(), llvm.DoubleType()):
            return builder.fsub(left, right)
        case _:
            raise TypeError(f"Cannot add {left.type} and {right.type}")


@ir_to_llvm_expression.register(Multiply)
def ir_to_llvm_multiply(
    self: Multiply, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left = ir_to_llvm_expression(self.left, builder, locals)
    right = ir_to_llvm_expression(self.right, builder, locals)
    match left.type, right.type:
        case (llvm.IntType(), llvm.IntType()):
            return builder.mul(left, right)
        case (llvm.IntType(), llvm.DoubleType()):
            left = builder.sitofp(left, llvm_float_type)
            return builder.fmul(left, right)
        case (llvm.DoubleType(), llvm.IntType()):
            right = builder.sitofp(right, llvm_float_type)
            return builder.fmul(left, right)
        case (llvm.DoubleType(), llvm.DoubleType()):
            return builder.fmul(left, right)
        case _:
            raise TypeError(f"Cannot add {left.type} and {right.type}")


@ir_to_llvm_expression.register(Equal)
def ir_to_llvm_equal(
    self: Equal, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        "==",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(NotEqual)
def ir_to_llvm_not_equal(
    self: NotEqual, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        "!=",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(GreaterThan)
def ir_to_llvm_greater_than(
    self: GreaterThan, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        ">",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(LessThan)
def ir_to_llvm_less_than(
    self: LessThan, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        "<",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(GreaterThanOrEqual)
def ir_to_llvm_greater_than_or_equal(
    self: GreaterThanOrEqual, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        ">=",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(LessThanOrEqual)
def ir_to_llvm_less_than_or_equal(
    self: LessThanOrEqual, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    return builder.icmp_signed(
        "<=",
        ir_to_llvm_expression(self.left, builder, locals),
        ir_to_llvm_expression(self.right, builder, locals),
    )


@ir_to_llvm_expression.register(And)
def ir_to_llvm_and(
    self: And, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left_block = builder.block
    right_block = builder.append_basic_block()
    end_block = builder.append_basic_block()

    left = ir_to_llvm_expression(self.left, builder, locals)

    builder.cbranch(left, right_block, end_block)

    builder.position_at_end(right_block)
    right = ir_to_llvm_expression(self.right, builder, locals)
    builder.branch(end_block)

    builder.position_at_end(end_block)
    phi = builder.phi(llvm_boolean_type)
    phi.add_incoming(llvm.Constant(llvm_boolean_type, 0), left_block)
    phi.add_incoming(right, right_block)

    return phi


@ir_to_llvm_expression.register(Or)
def ir_to_llvm_or(self: Or, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]) -> llvm.Value:
    left_block = builder.block
    right_block = builder.append_basic_block()
    end_block = builder.append_basic_block()

    left = ir_to_llvm_expression(self.left, builder, locals)

    builder.cbranch(left, end_block, right_block)

    builder.position_at_end(right_block)
    right = ir_to_llvm_expression(self.right, builder, locals)
    builder.branch(end_block)

    builder.position_at_end(end_block)
    phi = builder.phi(llvm_boolean_type)
    phi.add_incoming(llvm.Constant(llvm_boolean_type, 1), left_block)
    phi.add_incoming(right, right_block)

    return phi


@ir_to_llvm_expression.register(Max)
def ir_to_llvm_max(
    self: Max, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left = ir_to_llvm_expression(self.left, builder, locals)
    right = ir_to_llvm_expression(self.right, builder, locals)
    condition = builder.icmp_signed(">", left, right)
    return builder.select(condition, left, right)


@ir_to_llvm_expression.register(Min)
def ir_to_llvm_min(
    self: Min, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    left = ir_to_llvm_expression(self.left, builder, locals)
    right = ir_to_llvm_expression(self.right, builder, locals)
    condition = builder.icmp_signed("<", left, right)
    return builder.select(condition, left, right)


@ir_to_llvm_expression.register(BooleanToInteger)
def ir_to_llvm_boolean_to_integer(
    self: BooleanToInteger, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    expression = ir_to_llvm_expression(self.expression, builder, locals)
    return builder.zext(expression, llvm_integer_type)


@ir_to_llvm_expression.register(ArrayAllocate)
def ir_to_llvm_array_allocate(
    self: ArrayAllocate, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    element_size = llvm.Constant(
        llvm_integer_type, type_to_llvm(self.element_type).get_abi_size(target_machine.target_data)
    )
    n_elements = ir_to_llvm_expression(self.n_elements, builder, locals)
    memory_size = builder.zext(builder.mul(element_size, n_elements), llvm_size_type)
    memory_pointer = builder.call(locals["malloc"], [memory_size])
    array_pointer = builder.bitcast(memory_pointer, type_to_llvm(self.element_type).as_pointer())
    return array_pointer


@ir_to_llvm_expression.register(ArrayReallocate)
def ir_to_llvm_array_reallocate(
    self: ArrayReallocate, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    old_array_pointer = ir_to_llvm_expression(self.old, builder, locals)
    old_memory_pointer = builder.bitcast(old_array_pointer, llvm.IntType(8).as_pointer())
    element_size = llvm.Constant(
        llvm_integer_type, type_to_llvm(self.element_type).get_abi_size(target_machine.target_data)
    )
    n_elements = ir_to_llvm_expression(self.n_elements, builder, locals)
    memory_size = builder.zext(builder.mul(element_size, n_elements), llvm_size_type)
    memory_pointer = builder.call(locals["realloc"], [old_memory_pointer, memory_size])
    array_pointer = builder.bitcast(memory_pointer, type_to_llvm(self.element_type).as_pointer())
    return array_pointer


def ir_to_llvm_declaration(
    self: Declaration, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> llvm.Value:
    # All declarations are hoisted in LLVM, so this merely fetches the previously declared variable
    return locals[self.name.name]


@singledispatch
def ir_to_llvm_statement(
    self: Statement, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    raise NotImplementedError(f"ir_to_llvm_statement not implemented for {type(self)}: {self}")


@ir_to_llvm_statement.register(Expression)
def convert_expression_to_statement(
    self: Expression, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    # An expression can also be a statement; run it here and throw away the result
    _ = ir_to_llvm_expression(self)


@ir_to_llvm_statement.register(Declaration)
def convert_declaration_to_statement(
    self: Declaration, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    # A declaration can also be a statement; declare the variable and throw away the result
    _ = ir_to_llvm_declaration(self, builder, locals)


@ir_to_llvm_statement.register(Assignment)
def ir_to_llvm_assignment(
    self: Assignment, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    value = ir_to_llvm_expression(self.value, builder, locals)
    target = get_element_pointer(self.target, builder, locals)
    match value.type, target.type.pointee:
        case (llvm.IntType(), llvm.DoubleType()):
            value = builder.sitofp(value, llvm_float_type)
        case _:
            pass
    builder.store(value, target)


@ir_to_llvm_statement.register(DeclarationAssignment)
def ir_to_llvm_declaration_assignment(
    self: DeclarationAssignment, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    value = ir_to_llvm_expression(self.value, builder, locals)
    target = ir_to_llvm_declaration(self.target, builder, locals)
    match value.type, target.type:
        case (llvm.IntType(), llvm.DoubleType()):
            value = builder.sitofp(value, llvm_float_type)
        case _:
            pass
    builder.store(value, target)


@ir_to_llvm_statement.register(Block)
def ir_to_llvm_block(self: Block, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]) -> None:
    if self.comment is not None:
        # Add comment if it there is one
        builder.comment(self.comment)

    for statement in self.statements:
        ir_to_llvm_statement(statement, builder, locals)


@ir_to_llvm_statement.register(Branch)
def ir_to_llvm_branch(
    self: Branch, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    condition = ir_to_llvm_expression(self.condition, builder, locals)
    with builder.if_else(condition) as (then, otherwise):
        with then:
            ir_to_llvm_statement(self.if_true, builder, locals)
        with otherwise:
            ir_to_llvm_statement(self.if_false, builder, locals)


@ir_to_llvm_statement.register(Loop)
def ir_to_llvm_loop(self: Loop, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]) -> None:
    condition_block = builder.append_basic_block()
    body_block = builder.append_basic_block()
    end_block = builder.append_basic_block()

    builder.branch(condition_block)

    builder.position_at_end(condition_block)
    condition = ir_to_llvm_expression(self.condition, builder, locals)
    builder.cbranch(condition, body_block, end_block)

    builder.position_at_end(body_block)
    ir_to_llvm_statement(self.body, builder, locals)
    builder.branch(condition_block)

    builder.position_at_end(end_block)


@ir_to_llvm_statement.register(Return)
def ir_to_llvm_return(
    self: Return, builder: llvm.IRBuilder, locals: dict[str, llvm.Value]
) -> None:
    builder.ret(ir_to_llvm_expression(self.value, builder, locals))


def ir_to_llvm_function_definition(
    self: FunctionDefinition, module: llvm.Module, functions: dict[str, llvm.Function]
) -> None:
    return_type = type_to_llvm(self.return_type)
    argument_types = [type_to_llvm(parameter.type) for parameter in self.parameters]
    function_type = llvm.FunctionType(return_type, argument_types)
    function = llvm.Function(module, function_type, name=self.name.name)

    body = llvm.IRBuilder(function.append_basic_block())
    locals = {}

    # Assign each function parameter to a local variable
    for parameter, llvm_parameter in zip(self.parameters, function.args, strict=False):
        variable = body.alloca(type_to_llvm(parameter.type), name=parameter.name.name)
        body.store(llvm_parameter, variable)
        locals[parameter.name.name] = variable

    # Hoist all declarations in the function body
    for name, type in hoist_declarations(self).items():
        variable = body.alloca(type_to_llvm(type), name=name)
        locals[name] = variable

    ir_to_llvm_statement(self.body, body, locals | functions)


def ir_to_llvm(self: Module) -> llvm.Module:
    module = llvm.Module()

    # Declare malloc
    malloc_signature = llvm.FunctionType(llvm.IntType(8).as_pointer(), [llvm_size_type])
    malloc = llvm.Function(module, malloc_signature, name="malloc")

    # Declare realloc
    realloc_signature = llvm.FunctionType(
        llvm.IntType(8).as_pointer(), [llvm.IntType(8).as_pointer(), llvm_size_type]
    )
    realloc = llvm.Function(module, realloc_signature, name="realloc")

    # This is not compatible with defined functions ever being called
    functions = {"malloc": malloc, "realloc": realloc}

    for function in self.definitions:
        ir_to_llvm_function_definition(function, module, functions)

    return module
