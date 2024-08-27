from hypothesis import given

from tensora.compile import BroadcastTargetIndexError, TensorMethod
from tensora.desugar import DiagonalAccessError, NoKernelFoundError

from .strategies import problem_and_tensors


@given(problem_and_tensors())
def test_generate_cannot_crash(problem_inputs):
    problem, input_tensors = problem_inputs

    try:
        method = TensorMethod(problem)
    except (BroadcastTargetIndexError, DiagonalAccessError, NoKernelFoundError):
        return

    _ = method(**input_tensors)
