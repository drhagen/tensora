__all__ = ["evaluate", "evaluate_cffi", "evaluate_tensora", "tensor_method"]

from functools import lru_cache

from returns.functions import raise_exception

from ..expression import parse_assignment
from ..format import parse_format
from ..problem import Problem, make_problem
from ..tensor import Tensor
from ._tensor_method import BackendCompiler, TensorMethod


@lru_cache()
def cachable_tensor_method(problem: Problem, backend: BackendCompiler) -> TensorMethod:
    return TensorMethod(problem, backend=backend)


def tensor_method(
    assignment: str,
    formats: dict[str, str],
    backend: BackendCompiler = BackendCompiler.llvm,
) -> TensorMethod:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    parsed_formats = {
        name: parse_format(format).alt(raise_exception).unwrap()
        for name, format in formats.items()
    }

    problem = make_problem(parsed_assignment, parsed_formats).alt(raise_exception).unwrap()

    return cachable_tensor_method(problem, backend)


def evaluate_cffi(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, BackendCompiler.cffi)

    return function(**inputs)


def evaluate_tensora(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, BackendCompiler.llvm)

    return function(**inputs)


def evaluate(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    return evaluate_tensora(assignment, output_format, **inputs)
