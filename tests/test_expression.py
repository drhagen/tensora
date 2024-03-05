import pytest

from tensora.expression import (
    InconsistentDimensionsError,
    MutatingAssignmentError,
    NameConflictError,
    parse_assignment,
)
from tensora.expression.ast import *

assignment_strings = [
    (
        "A(i) = B(i,j) * C(j)",
        Assignment(Tensor("A", ("i",)), Multiply(Tensor("B", ("i", "j")), Tensor("C", ("j",)))),
    ),
    (
        "ab(i) = a(i) + b(i)",
        Assignment(Tensor("ab", ("i",)), Add(Tensor("a", ("i",)), Tensor("b", ("i",)))),
    ),
    (
        "D(i) = A(i) - B(i)",
        Assignment(Tensor("D", ("i",)), Subtract(Tensor("A", ("i",)), Tensor("B", ("i",)))),
    ),
    (
        "B2(i) = 2.0 * B(i)",
        Assignment(Tensor("B2", ("i",)), Multiply(Float(2.0), Tensor("B", ("i",)))),
    ),
    (
        "ab2(i) = 2.0 * (a(i) + b(i))",
        Assignment(
            Tensor("ab2", ("i",)),
            Multiply(Float(2.0), Add(Tensor("a", ("i",)), Tensor("b", ("i",)))),
        ),
    ),
    (
        "ab2(i) = (a(i) + b(i)) * 2.0",
        Assignment(
            Tensor("ab2", ("i",)),
            Multiply(Add(Tensor("a", ("i",)), Tensor("b", ("i",))), Float(2.0)),
        ),
    ),
]


@pytest.mark.parametrize(("string", "assignment"), assignment_strings)
def test_assignment_parsing(string, assignment):
    actual = parse_assignment(string).unwrap()
    assert actual == assignment


@pytest.mark.parametrize(("string", "assignment"), assignment_strings)
def test_assignment_deparsing(string, assignment):
    deparsed = assignment.deparse()
    assert deparsed == string


def test_mutating_assignment():
    assert isinstance(parse_assignment("A(i) = A(i) + 1").failure(), MutatingAssignmentError)


@pytest.mark.parametrize(
    "assignment",
    [
        "A(i) = B(i) + B(i,j)",
        "A(i) = B(i) - B()",
        "A(i) = B(i,j) * B(k,l,m)",
        "A(i) = B(i,j) + C(j,k) + (B(k) * D(k))",
    ],
)
def test_inconsistent_variable_size(assignment):
    assert isinstance(parse_assignment(assignment).failure(), InconsistentDimensionsError)


@pytest.mark.parametrize(
    "assignment",
    [
        "A(i) = B(B)",
        "A(i) = B(i,j) * C(j,B)",
        "A(i) = C(j,B) * B(i,j)",
        "A(A) = B(i)",
        "A(i) = B(A)",
        "A(B) = B(i)",
    ],
)
def test_name_conflict(assignment):
    assert isinstance(parse_assignment(assignment).failure(), NameConflictError)


def parse(string):
    return parse_assignment(string).unwrap()


def test_assignment_to_string():
    string = "A(i) = 2 * B(i,j) * (C(j) + D(j))"
    assert str(parse(string)) == string


@pytest.mark.parametrize(
    ("string", "output"),
    [
        (
            "y(i) = 0.5 * (b() - a()) * (x1(i,j) + x2(i,j)) * z(j)",
            {"y": 1, "b": 0, "a": 0, "x1": 2, "x2": 2, "z": 1},
        ),
        ("B2(i,k) = B(i,j) * B(j,k)", {"B2": 2, "B": 2}),
    ],
)
def test_variable_order(string, output):
    assert parse(string).variable_orders() == output


@pytest.mark.parametrize(
    ("string", "output"),
    [
        (
            "y(i) = 0.5 * (b() - a()) * (x1(i,j) + x2(i,j)) * z(j)",
            {"i": {("y", 0), ("x1", 0), ("x2", 0)}, "j": {("x1", 1), ("x2", 1), ("z", 0)}},
        ),
        (
            "B2(i,k) = B(i,j) * B(j,k)",
            {"i": {("B2", 0), ("B", 0)}, "j": {("B", 1), ("B", 0)}, "k": {("B2", 1), ("B", 1)}},
        ),
        ("diagA2(i) = A(i,i) + A(i,i)", {"i": {("diagA2", 0), ("A", 0), ("A", 1)}}),
    ],
)
def test_index_participants(string, output):
    assert parse(string).index_participants() == output
