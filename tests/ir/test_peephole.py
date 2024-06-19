import pytest

from tensora.ir import peephole, peephole_function_definition, peephole_statement
from tensora.ir.ast import *
from tensora.ir.types import *

changed = [
    # Add zero
    (Add(IntegerLiteral(0), Variable("x")), Variable("x")),
    (Add(Variable("x"), IntegerLiteral(0)), Variable("x")),
    (Add(FloatLiteral(0.0), Variable("x")), Variable("x")),
    (Add(Variable("x"), FloatLiteral(0.0)), Variable("x")),
    # Minus zero
    (Subtract(Variable("x"), IntegerLiteral(0)), Variable("x")),
    (Subtract(Variable("x"), FloatLiteral(0.0)), Variable("x")),
    # Multiply 0
    (Multiply(IntegerLiteral(0), Variable("x")), IntegerLiteral(0)),
    (Multiply(Variable("x"), IntegerLiteral(0)), IntegerLiteral(0)),
    (Multiply(FloatLiteral(0.0), Variable("x")), FloatLiteral(0.0)),
    (Multiply(Variable("x"), FloatLiteral(0.0)), FloatLiteral(0.0)),
    # Multiply 1
    (Multiply(IntegerLiteral(1), Variable("x")), Variable("x")),
    (Multiply(Variable("x"), IntegerLiteral(1)), Variable("x")),
    (Multiply(FloatLiteral(1.0), Variable("x")), Variable("x")),
    (Multiply(Variable("x"), FloatLiteral(1.0)), Variable("x")),
    # Equals same
    (Equal(Variable("x"), Variable("x")), BooleanLiteral(True)),
    (LessThanOrEqual(Variable("x"), Variable("x")), BooleanLiteral(True)),
    (GreaterThanOrEqual(Variable("x"), Variable("x")), BooleanLiteral(True)),
    # Not equal same
    (NotEqual(Variable("x"), Variable("x")), BooleanLiteral(False)),
    (LessThan(Variable("x"), Variable("x")), BooleanLiteral(False)),
    (GreaterThan(Variable("x"), Variable("x")), BooleanLiteral(False)),
    # And true
    (And(BooleanLiteral(True), Variable("x")), Variable("x")),
    (And(Variable("x"), BooleanLiteral(True)), Variable("x")),
    # And false
    (And(BooleanLiteral(False), Variable("x")), BooleanLiteral(False)),
    (And(Variable("x"), BooleanLiteral(False)), BooleanLiteral(False)),
    # Or true
    (Or(BooleanLiteral(True), Variable("x")), BooleanLiteral(True)),
    (Or(Variable("x"), BooleanLiteral(True)), BooleanLiteral(True)),
    # Or false
    (Or(BooleanLiteral(False), Variable("x")), Variable("x")),
    (Or(Variable("x"), BooleanLiteral(False)), Variable("x")),
    # Boolean cast constant
    (BooleanToInteger(BooleanLiteral(False)), IntegerLiteral(0)),
    (BooleanToInteger(BooleanLiteral(True)), IntegerLiteral(1)),
    # Branch true
    (Branch(BooleanLiteral(True), Variable("x"), Variable("y")), Variable("x")),
    # Branch false
    (Branch(BooleanLiteral(False), Variable("x"), Variable("y")), Variable("y")),
    # Loop false
    (Loop(BooleanLiteral(False), Variable("x")), Block([])),
    # Empty block
    (Block([Block([], "comment"), Return(IntegerLiteral(0))]), Block([Return(IntegerLiteral(0))])),
    (Branch(Variable("t"), Block([], "comment1"), Block([], "comment2")), Block([])),
    (Loop(Variable("t"), Block([], "comment")), Block([])),
    # Redundant assignment
    (Assignment(Variable("x"), Variable("x")), Block([])),
    (
        Assignment(
            ArrayIndex(Variable("x"), IntegerLiteral(0)),
            ArrayIndex(Variable("x"), IntegerLiteral(0)),
        ),
        Block([]),
    ),
    # Pass through
    (
        ArrayIndex(
            ArrayIndex(Variable("x"), Add(IntegerLiteral(0), Variable("i"))), Variable("j")
        ),
        ArrayIndex(ArrayIndex(Variable("x"), Variable("i")), Variable("j")),
    ),
    (
        ArrayIndex(Variable("x"), Add(IntegerLiteral(0), Variable("i"))),
        ArrayIndex(Variable("x"), Variable("i")),
    ),
    (
        AttributeAccess(ArrayIndex(Variable("x"), Add(IntegerLiteral(0), Variable("i"))), "modes"),
        AttributeAccess(ArrayIndex(Variable("x"), Variable("i")), "modes"),
    ),
    (
        BooleanToInteger(And(BooleanLiteral(True), Variable("t"))),
        BooleanToInteger(Variable("t")),
    ),
    (
        ArrayAllocate(float, Add(IntegerLiteral(0), Variable("x"))),
        ArrayAllocate(float, Variable("x")),
    ),
    (
        ArrayReallocate(Variable("x"), float, Add(IntegerLiteral(0), Variable("y"))),
        ArrayReallocate(Variable("x"), float, Variable("y")),
    ),
    (
        Assignment(Variable("x"), Add(IntegerLiteral(0), Variable("y"))),
        Assignment(Variable("x"), Variable("y")),
    ),
    (
        Assignment(
            ArrayIndex(Variable("x"), Add(IntegerLiteral(0), Variable("i"))), Variable("y")
        ),
        Assignment(ArrayIndex(Variable("x"), Variable("i")), Variable("y")),
    ),
    (
        DeclarationAssignment(
            Declaration(Variable("x"), float), Add(FloatLiteral(0.0), Variable("y"))
        ),
        DeclarationAssignment(Declaration(Variable("x"), float), Variable("y")),
    ),
    (
        Block(
            [
                Assignment(
                    Variable("x"), Add(Add(IntegerLiteral(0), Variable("x")), IntegerLiteral(1))
                )
            ],
            "test block",
        ),
        Block([Assignment(Variable("x"), Add(Variable("x"), IntegerLiteral(1)))], "test block"),
    ),
    (
        Branch(Variable("t"), Variable("x"), Add(IntegerLiteral(0), Variable("y"))),
        Branch(Variable("t"), Variable("x"), Variable("y")),
    ),
    (
        Branch(
            Variable("t"),
            Add(IntegerLiteral(0), Variable("x")),
            Add(IntegerLiteral(0), Variable("y")),
        ),
        Branch(Variable("t"), Variable("x"), Variable("y")),
    ),
    (
        Branch(
            And(BooleanLiteral(True), Variable("t")),
            Variable("x"),
            Add(IntegerLiteral(0), Variable("y")),
        ),
        Branch(Variable("t"), Variable("x"), Variable("y")),
    ),
    (
        Loop(Variable("t"), Add(IntegerLiteral(0), Variable("x"))),
        Loop(Variable("t"), Variable("x")),
    ),
    (
        Loop(And(BooleanLiteral(True), Variable("t")), Variable("x")),
        Loop(Variable("t"), Variable("x")),
    ),
    (
        Return(Add(IntegerLiteral(0), Variable("x"))),
        Return(Variable("x")),
    ),
]


@pytest.mark.parametrize(("before", "after"), changed)
def test_peephole_statement(before: Statement, after: Statement):
    assert peephole_statement(before) == after


left_right_classes = [
    Add,
    Subtract,
    Multiply,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Max,
    Min,
]


@pytest.mark.parametrize("cls", left_right_classes)
def test_pass_through_left_right(cls):
    left = Add(IntegerLiteral(0), Variable("x"))
    right = Add(IntegerLiteral(0), Variable("y"))
    expected = cls(Variable("x"), Variable("y"))
    assert peephole_statement(cls(left, Variable("y"))) == expected
    assert peephole_statement(cls(Variable("x"), right)) == expected


unchanged = [
    Subtract(IntegerLiteral(0), Variable("x")),
    Subtract(FloatLiteral(0.0), Variable("x")),
    Equal(Variable("x"), Variable("y")),
    LessThanOrEqual(Variable("x"), Variable("y")),
    GreaterThanOrEqual(Variable("x"), Variable("y")),
    NotEqual(Variable("x"), Variable("y")),
    LessThan(Variable("x"), Variable("y")),
    GreaterThan(Variable("x"), Variable("y")),
    Loop(BooleanLiteral(True), Variable("x")),
    Assignment(Variable("x"), ArrayIndex(ArrayIndex(Variable("y"), Variable("i")), Variable("j"))),
    Declaration(Variable("x"), float),
]


@pytest.mark.parametrize("input", unchanged)
def test_peephole_statement_noop(input: Statement):
    assert peephole_statement(input) == input


def test_peephole_function_definition():
    function = FunctionDefinition(
        Variable("f"),
        [Declaration(Variable("x"), tensor)],
        integer,
        Return(Multiply(IntegerLiteral(0), IntegerLiteral(1))),
    )
    expected = FunctionDefinition(
        Variable("f"),
        [Declaration(Variable("x"), tensor)],
        integer,
        Return(IntegerLiteral(0)),
    )
    assert peephole_function_definition(function) == expected


def test_peephole_module():
    input_function = FunctionDefinition(
        Variable("f"),
        [Declaration(Variable("x"), tensor)],
        integer,
        Return(Multiply(IntegerLiteral(0), IntegerLiteral(1))),
    )
    expected_function = FunctionDefinition(
        Variable("f"),
        [Declaration(Variable("x"), tensor)],
        integer,
        Return(IntegerLiteral(0)),
    )

    module = Module([input_function, input_function])
    expected = Module([expected_function, expected_function])
    assert peephole(module) == expected
