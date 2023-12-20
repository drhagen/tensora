from hypothesis import given

from tensora.desugar import DiagonalAccessError, NoKernelFoundError
from tensora.function import PureTensorMethod

from .strategies import problem_and_tensors


@given(problem_and_tensors())
def test_generate_cannot_crash(problem_inputs):
    problem, input_tensors = problem_inputs

    input_formats = {
        name: format
        for name, format in problem.formats.items()
        if name != problem.assignment.target.name
    }
    output_format = problem.formats[problem.assignment.target.name]
    try:
        method = PureTensorMethod(problem.assignment, input_formats, output_format)
    except (DiagonalAccessError, NoKernelFoundError):
        return

    _ = method(**input_tensors)
