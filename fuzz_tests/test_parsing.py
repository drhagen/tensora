import hypothesis.strategies as st
from hypothesis import given
from parsita import ParseError
from returns.result import Failure, Success

from tensora.expression import (
    InconsistentDimensionsError,
    MutatingAssignmentError,
    NameConflictError,
    ast,
    parse_assignment,
)
from tensora.format import Format, InvalidModeOrderingError, parse_format

from .strategies import assignments, formats


@given(st.text())
def test_format_parsing_cannot_crash(string):
    match parse_format(string):
        case Success(Format()):
            pass
        case Failure(ParseError() | InvalidModeOrderingError()):
            pass
        case _:
            raise RuntimeError("Unexpected result")


@given(formats())
def test_format_parsing_round_trips(format):
    text = format.deparse()
    new_format = parse_format(text).unwrap()
    assert format == new_format


@given(st.text())
def test_expression_parsing_cannot_crash(string):
    match parse_assignment(string):
        case Success(ast.Assignment()):
            pass
        case Failure(
            ParseError()
            | MutatingAssignmentError()
            | InconsistentDimensionsError()
            | NameConflictError()
        ):
            pass
        case _:
            raise RuntimeError("Unexpected result")


@given(assignments())
def test_expression_parsing_round_trips(assignment):
    text = assignment.deparse()
    new_assignment = parse_assignment(text).unwrap()
    assert assignment == new_assignment
