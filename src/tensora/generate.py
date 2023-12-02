__all__ = [
    "TensorCompiler",
    "generate_c_code",
    "generate_c_code_tensora",
    "generate_c_code_taco",
    "TacoError",
]

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from returns.result import Failure, Result, Success

from .desugar import DiagonalAccessError, NoKernelFoundError
from .kernel_type import KernelType
from .problem import Problem


class TensorCompiler(str, Enum):
    # Python 3.10 does not support StrEnum, so do it manually
    taco = "taco"
    tensora = "tensora"

    def __str__(self) -> str:
        return self.name


def generate_c_code(
    problem: Problem, kernel_types: list[KernelType], tensor_compiler: TensorCompiler
) -> Result[str, Exception]:
    match tensor_compiler:
        case TensorCompiler.tensora:
            return generate_c_code_tensora(problem, kernel_types)
        case TensorCompiler.taco:
            return generate_c_code_taco(problem, kernel_types)


def generate_c_code_tensora(
    problem: Problem, kernel_types: list[KernelType]
) -> Result[str, DiagonalAccessError | NoKernelFoundError]:
    from .codegen import ir_to_c
    from .desugar import best_algorithm, desugar_assignment, index_dimensions, to_identifiable
    from .ir import SourceBuilder, peephole
    from .iteration_graph import Definition, generate_ir

    formats = problem.formats

    desugar = desugar_assignment(problem.assignment)

    output_variable = to_identifiable(desugar.target, formats)

    definition = Definition(output_variable, formats, index_dimensions(desugar))

    match best_algorithm(desugar, formats):
        case Failure() as result:
            return result
        case Success(graph):
            pass
        case _:
            raise NotImplementedError()

    ir = SourceBuilder()
    for kernel_type in kernel_types:
        ir.append(generate_ir(definition, graph, kernel_type).finalize())

    return Success(ir_to_c(peephole(ir.finalize())))


@dataclass(frozen=True, slots=True)
class TacoError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


taco_binary = Path(__file__).parent.joinpath("taco/bin/taco")


def generate_c_code_taco(
    problem: Problem, kernel_types: list[KernelType]
) -> Result[str, Exception]:
    import subprocess

    from .expression import deparse_to_taco

    formats = problem.formats

    expression_string = deparse_to_taco(problem.assignment)
    format_string_arguments = []
    for name, format in formats.items():
        if format is not None and format.order != 0:  # Taco does not like formats for scalars
            mode_string = "".join(mode.character for mode in format.modes)
            ordering_string = ",".join(map(str, format.ordering))
            format_string_arguments.append(f"-f={name}:{mode_string}:{ordering_string}")

    kernel_type_arguments = []
    for kernel_type in kernel_types:
        if kernel_type == KernelType.evaluate:
            kernel_type_arguments.append("-print-evaluate")
        elif kernel_type == KernelType.compute:
            kernel_type_arguments.append("-print-compute")
        elif kernel_type == KernelType.assemble:
            kernel_type_arguments.append("-print-assembly")

    # Call taco to write the kernels to standard out
    result = subprocess.run(
        [taco_binary, expression_string, "-print-nocolor"]
        + kernel_type_arguments
        + format_string_arguments,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return Failure(TacoError(result.stderr))

    return Success(result.stdout)
