__all__ = ["generate_c_code_from_parsed"]

from .codegen import ir_to_c
from .desugar import best_algorithm, desugar_assignment, index_dimensions, to_identifiable
from .expression.ast import Assignment
from .format import Format
from .ir import SourceBuilder, peephole
from .iteration_graph import Definition, generate_ir
from .kernel_type import KernelType


def generate_c_code_from_parsed(
    assignment: Assignment,
    formats: dict[str, Format],
    kernel_types: list[KernelType] = [KernelType.evaluate],
) -> str:
    desugar = desugar_assignment(assignment)

    output_variable = to_identifiable(desugar.target, formats)

    problem = Definition(output_variable, formats, index_dimensions(desugar))

    graph = best_algorithm(desugar, formats)

    ir = SourceBuilder()
    for kernel_type in kernel_types:
        ir.append(generate_ir(problem, graph, kernel_type).finalize())

    return ir_to_c(peephole(ir.finalize()))
