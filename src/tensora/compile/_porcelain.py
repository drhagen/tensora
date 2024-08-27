__all__ = ["tensor_method", "evaluate", "evaluate_taco", "evaluate_cffi", "evaluate_tensora"]

from functools import lru_cache

from returns.functions import raise_exception

from ..expression import parse_assignment
from ..format import parse_format
from ..generate import TensorCompiler
from ..problem import Problem, make_problem
from ..tensor import Tensor
from ._tensor_method import BackendCompiler, TensorMethod


@lru_cache()
def cachable_tensor_method(
    problem: Problem, compiler: TensorCompiler, backend: BackendCompiler
) -> TensorMethod:
    return TensorMethod(problem, compiler, backend)


def tensor_method(
    assignment: str,
    formats: dict[str, str],
    compiler: TensorCompiler = TensorCompiler.tensora,
    backend: BackendCompiler | None = None,
) -> TensorMethod:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    parsed_formats = {
        name: parse_format(format).alt(raise_exception).unwrap()
        for name, format in formats.items()
    }

    problem = make_problem(parsed_assignment, parsed_formats).alt(raise_exception).unwrap()

    if backend is None:
        if compiler == TensorCompiler.tensora:
            backend = BackendCompiler.llvm
        else:
            backend = BackendCompiler.cffi

    return cachable_tensor_method(problem, compiler, backend)


def evaluate_taco(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, TensorCompiler.taco, BackendCompiler.cffi)

    return function(**inputs)


def evaluate_cffi(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, TensorCompiler.tensora, BackendCompiler.cffi)

    return function(**inputs)


def evaluate_tensora(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, TensorCompiler.tensora, BackendCompiler.llvm)

    return function(**inputs)


def evaluate(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    return evaluate_tensora(assignment, output_format, **inputs)
