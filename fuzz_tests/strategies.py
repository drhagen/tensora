import hypothesis.strategies as st
from hypothesis import assume

from tensora import Tensor
from tensora.expression import (
    InconsistentDimensionsError,
    MutatingAssignmentError,
    NameConflictError,
    ast,
)
from tensora.format import Format, Mode
from tensora.problem import Problem

names = st.from_regex(r"[A-Za-z][A-Za-z0-9]*", fullmatch=True)
variables = st.builds(
    ast.Tensor, name=names, indexes=st.builds(tuple, st.lists(names, max_size=4))
)
expressions = st.deferred(
    lambda: st.builds(ast.Integer, st.integers(min_value=0, max_value=2**16))
    | st.builds(ast.Float, st.floats(min_value=0, allow_infinity=False, allow_nan=False))
    | variables
    | adds
    | subtracts
    | multiplies
)
adds = st.builds(ast.Add, expressions, expressions)
subtracts = st.builds(ast.Subtract, expressions, expressions)
multiplies = st.builds(ast.Multiply, expressions, expressions)


@st.composite
def assignments(draw):
    target = draw(variables)
    expression = draw(expressions)
    try:
        return ast.Assignment(target, expression)
    except (InconsistentDimensionsError, MutatingAssignmentError, NameConflictError):
        assume(False)


modes = st.sampled_from(Mode)


@st.composite
def formats(draw, orders=st.integers(min_value=0, max_value=4)) -> Format:
    order = draw(orders)
    format_modes = tuple(draw(st.lists(modes, min_size=order, max_size=order)))
    format_mode_ordering = tuple(draw(st.permutations(range(order))))
    return Format(format_modes, format_mode_ordering)


@st.composite
def problems(draw) -> Problem:
    assignment = draw(assignments())

    problem_formats = {
        name: draw(formats(st.just(order))) for name, order in assignment.variable_orders().items()
    }

    return Problem(assignment, problem_formats)


@st.composite
def tensors(draw, format: Format, dimensions: tuple[int, ...] | None = None) -> Tensor:
    if dimensions is None:
        dimensions = draw(st.lists(st.integers(), min_size=format.order, max_size=format.order))

    if any(dim == 0 for dim in dimensions):
        # Hypothesis does not like being forced to draw empty dictionaries by giving it an
        # empty set of possible keys
        dok = {}
    else:
        dok = draw(
            st.dictionaries(
                st.tuples(*[st.integers(min_value=0, max_value=dim - 1) for dim in dimensions]),
                st.floats(allow_infinity=False, allow_nan=False),
            )
        )

    return Tensor.from_dok(dok, format=format, dimensions=dimensions)


@st.composite
def problem_and_tensors(draw):
    problem: Problem = draw(problems())

    # Indexes over the same dimension must have the same size.
    # Dimensions sharing an index must have the same size.
    participant_sizes: dict[tuple[str, int], int] = {}
    index_sizes: dict[str, int] = {}
    for index, participants in problem.assignment.index_participants().items():
        for participant in participants:
            if participant in participant_sizes:
                index_sizes[index] = participant_sizes[participant]
            elif index in index_sizes:
                participant_sizes[participant] = index_sizes[index]
            else:
                # This number can conspire with the number of dimensions to
                # require a LOT of memory.
                size = draw(st.integers(min_value=0, max_value=64))
                participant_sizes[participant] = size
                index_sizes[index] = size

    input_tensors = {}
    for name, variable in problem.assignment.expression.variables().items():
        format = problem.formats[name]
        dimensions = tuple(index_sizes[index] for index in variable[0].indexes)
        input_tensors[name] = draw(tensors(format, dimensions))

    return (problem, input_tensors)
