from textwrap import dedent

import pytest

from tensora.codegen import ir_to_c, ir_to_c_function_definition, ir_to_c_statement
from tensora.ir.ast import *
from tensora.ir.types import *


def clean(string: str) -> str:
    return dedent(string).strip()


single_lines = [
    # Assignables and literals
    (Variable("x"), "x"),
    (ArrayIndex(Variable("x"), Variable("i")), "x[i]"),
    (ArrayIndex(ArrayIndex(Variable("x"), Variable("i")), Variable("j")), "x[i][j]"),
    (AttributeAccess(Variable("x"), "modes"), "x->modes"),
    (IntegerLiteral(2), "2"),
    (FloatLiteral(1.0), "1.0"),
    (BooleanLiteral(False), "0"),
    (BooleanLiteral(True), "1"),
    # Add
    (Add(Variable("x"), Variable("y")), "x + y"),
    (Add(Add(Variable("x"), Variable("y")), Variable("z")), "x + y + z"),
    (Add(Variable("x"), Add(Variable("y"), Variable("z"))), "x + y + z"),
    (Add(Subtract(Variable("x"), Variable("y")), Variable("z")), "x - y + z"),
    (Add(Variable("x"), Subtract(Variable("y"), Variable("z"))), "x + y - z"),
    (Add(Multiply(Variable("x"), Variable("y")), Variable("z")), "x * y + z"),
    (Add(Variable("x"), Multiply(Variable("y"), Variable("z"))), "x + y * z"),
    # Subtract
    (Subtract(Variable("x"), Variable("y")), "x - y"),
    (Subtract(Add(Variable("x"), Variable("y")), Variable("z")), "x + y - z"),
    (Subtract(Variable("x"), Add(Variable("y"), Variable("z"))), "x - (y + z)"),
    (Subtract(Subtract(Variable("x"), Variable("y")), Variable("z")), "x - y - z"),
    (Subtract(Variable("x"), Subtract(Variable("y"), Variable("z"))), "x - (y - z)"),
    (Subtract(Multiply(Variable("x"), Variable("y")), Variable("z")), "x * y - z"),
    (Subtract(Variable("x"), Multiply(Variable("y"), Variable("z"))), "x - y * z"),
    # Multiply
    (Multiply(Variable("x"), Variable("y")), "x * y"),
    (Multiply(Multiply(Variable("x"), Variable("y")), Variable("z")), "x * y * z"),
    (Multiply(Variable("x"), Multiply(Variable("y"), Variable("z"))), "x * y * z"),
    (Multiply(Add(Variable("x"), Variable("y")), Variable("z")), "(x + y) * z"),
    (Multiply(Variable("x"), Add(Variable("y"), Variable("z"))), "x * (y + z)"),
    (Multiply(Subtract(Variable("x"), Variable("y")), Variable("z")), "(x - y) * z"),
    (Multiply(Variable("x"), Subtract(Variable("y"), Variable("z"))), "x * (y - z)"),
    # Comparison
    (Equal(Variable("x"), Variable("y")), "x == y"),
    (NotEqual(Variable("x"), Variable("y")), "x != y"),
    (GreaterThan(Variable("x"), Variable("y")), "x > y"),
    (LessThan(Variable("x"), Variable("y")), "x < y"),
    (GreaterThanOrEqual(Variable("x"), Variable("y")), "x >= y"),
    (LessThanOrEqual(Variable("x"), Variable("y")), "x <= y"),
    # Boolean operators
    (And(Variable("x"), Variable("y")), "x && y"),
    (Or(Variable("x"), Variable("y")), "x || y"),
    # Max/min
    (Max(Variable("x"), Variable("y")), "TACO_MAX(x, y)"),
    (Min(Variable("x"), Variable("y")), "TACO_MIN(x, y)"),
    (Max(Max(Variable("x"), Variable("y")), Variable("z")), "TACO_MAX(TACO_MAX(x, y), z)"),
    (Min(Min(Variable("x"), Variable("y")), Variable("z")), "TACO_MIN(TACO_MIN(x, y), z)"),
    # Cast
    (BooleanToInteger(Equal(Variable("x"), Variable("y"))), "(int32_t)(x == y)"),
    # Allocate
    (ArrayAllocate(integer, Variable("capacity")), "malloc(sizeof(int32_t) * capacity)"),
    (
        ArrayAllocate(float, Add(Variable("previous"), Variable("new"))),
        "malloc(sizeof(double) * (previous + new))",
    ),
    (
        ArrayReallocate(Variable("old"), integer, Variable("capacity")),
        "realloc(old, sizeof(int32_t) * capacity)",
    ),
    (
        ArrayReallocate(Variable("old"), float, Add(Variable("previous"), Variable("new"))),
        "realloc(old, sizeof(double) * (previous + new))",
    ),
    # Declaration and types
    (Declaration(Variable("x"), integer), "int32_t x"),
    (Declaration(Variable("x"), float), "double x"),
    (Declaration(Variable("x"), tensor), "taco_tensor_t x"),
    (Declaration(Variable("x"), Pointer(float)), "double* restrict x"),
    (Declaration(Variable("x"), Pointer(Pointer(integer))), "int32_t* restrict* restrict x"),
    (Declaration(Variable("x"), Array(float)), "double x[]"),
    (Declaration(Variable("x"), Array(Array(integer))), "int32_t x[][]"),
    (Declaration(Variable("x"), FixedArray(mode, 3)), "taco_mode_t x[3]"),
    (Declaration(Variable("x"), FixedArray(FixedArray(mode, 3), 2)), "taco_mode_t x[3][2]"),
    # Assignment
    (Assignment(Variable("x"), Variable("y")), "x = y"),
    (Assignment(Variable("x"), Add(Variable("x"), IntegerLiteral(1))), "x++"),
    (Assignment(Variable("x"), Add(Variable("x"), IntegerLiteral(2))), "x += 2"),
    (Assignment(Variable("x"), Subtract(Variable("x"), IntegerLiteral(1))), "x--"),
    (Assignment(Variable("x"), Subtract(Variable("x"), IntegerLiteral(2))), "x -= 2"),
    (Assignment(Variable("x"), Multiply(Variable("x"), IntegerLiteral(2))), "x *= 2"),
    (DeclarationAssignment(Declaration(Variable("x"), integer), Variable("y")), "int32_t x = y"),
    # Return
    (Return(IntegerLiteral(0)), "return 0"),
]


@pytest.mark.parametrize(("ast", "code"), single_lines)
def test_single_lines(ast: Expression, code: str):
    assert ir_to_c_statement(ast) == [code + ";"]


multiple_lines = [
    (
        Block(
            [
                DeclarationAssignment(Declaration(Variable("x"), integer), IntegerLiteral(0)),
                Assignment(Variable("x"), Add(Variable("x"), IntegerLiteral(2))),
            ]
        ),
        """
        int32_t x = 0;
        x += 2;
        """,
    ),
    (
        Block(
            [
                DeclarationAssignment(Declaration(Variable("x"), integer), IntegerLiteral(0)),
            ],
            "comment",
        ),
        """
        // comment
        int32_t x = 0;
        """,
    ),
    (
        Block(
            [
                Block(
                    [
                        DeclarationAssignment(
                            Declaration(Variable("x"), integer), IntegerLiteral(0)
                        ),
                    ],
                    "inner comment",
                ),
                DeclarationAssignment(Declaration(Variable("y"), integer), IntegerLiteral(0)),
            ],
        ),
        """
        // inner comment
        int32_t x = 0;

        int32_t y = 0;
        """,
    ),
    (
        Block(
            [
                DeclarationAssignment(Declaration(Variable("y"), integer), IntegerLiteral(0)),
                Block(
                    [
                        DeclarationAssignment(
                            Declaration(Variable("x"), integer), IntegerLiteral(0)
                        ),
                    ],
                    "inner comment",
                ),
            ],
        ),
        """
        int32_t y = 0;

        // inner comment
        int32_t x = 0;
        """,
    ),
    (
        Block(
            [
                Block(
                    [
                        DeclarationAssignment(
                            Declaration(Variable("x"), integer), IntegerLiteral(0)
                        ),
                    ],
                    "inner comment",
                ),
                DeclarationAssignment(Declaration(Variable("y"), integer), IntegerLiteral(0)),
                DeclarationAssignment(Declaration(Variable("z"), integer), IntegerLiteral(0)),
            ],
            "comment",
        ),
        """
        // comment

        // inner comment
        int32_t x = 0;

        int32_t y = 0;
        int32_t z = 0;
        """,
    ),
    (
        Branch(
            Equal(Variable("x"), Variable("y")),
            Assignment(Variable("z"), IntegerLiteral(0)),
            Assignment(Variable("z"), IntegerLiteral(1)),
        ),
        """
        if (x == y) {
          z = 0;
        } else {
          z = 1;
        }
        """,
    ),
    (
        Branch(
            Equal(Variable("x"), Variable("y")),
            Assignment(Variable("z"), IntegerLiteral(0)),
            Block([]),
        ),
        """
        if (x == y) {
          z = 0;
        }
        """,
    ),
    (
        Branch(
            Equal(Variable("x"), IntegerLiteral(0)),
            Assignment(Variable("y"), Variable("a")),
            Branch(
                Equal(Variable("x"), IntegerLiteral(1)),
                Assignment(Variable("y"), Variable("b")),
                Assignment(Variable("y"), Variable("c")),
            ),
        ),
        """
        if (x == 0) {
          y = a;
        } else if (x == 1) {
          y = b;
        } else {
          y = c;
        }
        """,
    ),
    (
        Branch(
            Equal(Variable("x"), IntegerLiteral(0)),
            Assignment(Variable("y"), Variable("a")),
            Branch(
                Equal(Variable("x"), IntegerLiteral(1)),
                Assignment(Variable("y"), Variable("b")),
                Branch(
                    Equal(Variable("x"), IntegerLiteral(2)),
                    Assignment(Variable("y"), Variable("c")),
                    Block([]),
                ),
            ),
        ),
        """
        if (x == 0) {
          y = a;
        } else if (x == 1) {
          y = b;
        } else if (x == 2) {
          y = c;
        }
        """,
    ),
    (
        Loop(
            LessThan(Variable("x"), Variable("y")),
            Assignment(Variable("x"), Add(Variable("x"), IntegerLiteral(1))),
        ),
        """
        while (x < y) {
          x++;
        }
        """,
    ),
    (
        Loop(
            LessThan(Variable("x"), Variable("y")),
            Block(
                [DeclarationAssignment(Declaration(Variable("x"), integer), IntegerLiteral(0))],
                "comment",
            ),
        ),
        """
        while (x < y) {
          // comment
          int32_t x = 0;
        }
        """,
    ),
]


@pytest.mark.parametrize(("ast", "code"), multiple_lines)
def test_multiple_lines(ast: Statement, code: str):
    assert "\n".join(ir_to_c_statement(ast)) == clean(code)


def test_function_definition():
    function = FunctionDefinition(
        Variable("double_x"),
        [Declaration(Variable("x"), integer)],
        integer,
        Return(Multiply(Variable("x"), IntegerLiteral(2))),
    )
    expected = """
        int32_t double_x(int32_t x) {
          return x * 2;
        }
        """
    assert ir_to_c_function_definition(function) == clean(expected)


def test_module():
    function = FunctionDefinition(
        Variable("double_x"),
        [Declaration(Variable("x"), integer)],
        integer,
        Return(Multiply(Variable("x"), IntegerLiteral(2))),
    )
    module = Module([function, function])
    expected = """
        int32_t double_x(int32_t x) {
          return x * 2;
        }

        int32_t double_x(int32_t x) {
          return x * 2;
        }
        """
    assert ir_to_c(module) == clean(expected)
