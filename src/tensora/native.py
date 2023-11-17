__all__ = ["generate_c_code", "KernelType"]

from .codegen import ir_to_c
from .desugar import desugar_assignment, index_dimensions, to_identifiable, to_iteration_graphs
from .expression import parse_assignment
from .format import parse_format
from .ir import peephole
from .iteration_graph import Definition, generate_ir
from .kernel_type import KernelType


def generate_c_code(assignment: str, formats: dict[str, str], kernel_type: KernelType) -> str:
    assignment_parsed = parse_assignment(assignment).unwrap()
    formats_parsed = {name: parse_format(format).unwrap() for name, format in formats.items()}

    desugar = desugar_assignment(assignment_parsed)

    output_variable = to_identifiable(desugar.target, formats_parsed)

    problem = Definition(output_variable, formats_parsed, index_dimensions(desugar))

    graph = next(to_iteration_graphs(desugar, formats_parsed))

    ir = generate_ir(problem, graph, kernel_type).finalize()

    return ir_to_c(peephole(ir))
