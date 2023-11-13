__all__ = ["generate_code", "KernelType"]

from typing import Dict

from .codegen import ir_to_c
from .desugar import desugar_assignment, to_identifiable, to_iteration_graph
from .expression import parse_assignment
from .format import parse_format
from .ir import peephole
from .iteration_graph import KernelType, Problem, generate_c_code


def generate_code(
    assignment: str, output_format: str, input_formats: Dict[str, str], kernel_type: KernelType
) -> str:
    assignment_parsed = parse_assignment(assignment).unwrap()
    input_formats_parsed = {
        name: parse_format(format).unwrap() for name, format in input_formats.items()
    }
    output_format_parsed = parse_format(output_format).unwrap()

    desugar = desugar_assignment(assignment_parsed)

    identifiable_assignment = to_identifiable(desugar, input_formats_parsed, output_format_parsed)

    graph = to_iteration_graph(desugar, input_formats_parsed, output_format_parsed)
    problem = Problem(identifiable_assignment, input_formats_parsed, output_format_parsed)

    ir = generate_c_code(problem, graph, kernel_type).finalize()

    return ir_to_c(peephole(ir))
