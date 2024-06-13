__all__ = ["generate_c_code_tensora"]

from returns.result import Failure, Result, Success

from ..codegen import ir_to_c
from ..desugar import (
    DiagonalAccessError,
    NoKernelFoundError,
    best_algorithm,
    desugar_assignment,
    index_dimensions,
    to_identifiable,
)
from ..ir import peephole
from ..ir.ast import Module
from ..iteration_graph import Definition, generate_ir
from ..kernel_type import KernelType
from ..problem import Problem


def generate_c_code_tensora(
    problem: Problem, kernel_types: list[KernelType]
) -> Result[str, DiagonalAccessError | NoKernelFoundError]:
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

    functions = [generate_ir(definition, graph, kernel_type) for kernel_type in kernel_types]
    module = Module(functions)

    return Success(ir_to_c(peephole(module)))
