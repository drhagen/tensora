__all__ = ["generate_c_code", "generate_c_code_from_parsed", "KernelType"]

from .codegen import ir_to_c
from .desugar import desugar_assignment, index_dimensions, to_identifiable, to_iteration_graphs
from .expression import parse_assignment
from .expression.ast import Assignment
from .format import Format, parse_format
from .ir import SourceBuilder, peephole
from .iteration_graph import Definition, generate_ir
from .kernel_type import KernelType


def generate_c_code(assignment: str, formats: dict[str, str], kernel_type: KernelType) -> str:
    assignment_parsed = parse_assignment(assignment).unwrap()
    formats_parsed = {name: parse_format(format).unwrap() for name, format in formats.items()}

    return generate_c_code_from_parsed(assignment_parsed, formats_parsed, kernel_type)


def generate_c_code_from_parsed(
    assignment: Assignment,
    formats: dict[str, Format],
    kernel_types: list[KernelType] = [KernelType.evaluate],
) -> str:
    desugar = desugar_assignment(assignment)

    output_variable = to_identifiable(desugar.target, formats)

    problem = Definition(output_variable, formats, index_dimensions(desugar))

    graph = next(to_iteration_graphs(desugar, formats))

    ir = SourceBuilder()
    for kernel_type in kernel_types:
        ir.append(generate_ir(problem, graph, kernel_type).finalize())

    return ir_to_c(peephole(ir.finalize()))
