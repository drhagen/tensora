__all__ = ["generate_code", "KernelType"]

from typing import Dict
from tensora.expression import parse_assignment
from tensora.format import parse_format
from tensora.desugar import desugar_assignment, to_identifiable, to_iteration_graph
from tensora.iteration_graph import KernelType, generate_c_code, Problem
from tensora.codegen import ast_to_c
from tensora.ir import peephole


def generate_code(assignment: str, output_format: str, input_formats: Dict[str, str], kernel_type: KernelType) -> str:
    assignment_parsed = parse_assignment(assignment).unwrap()
    input_formats_parsed = {name: parse_format(format).unwrap() for name, format in input_formats.items()}
    output_format_parsed = parse_format(output_format).unwrap()

    desugar = desugar_assignment(assignment_parsed)

    identifiable_assignment = to_identifiable(desugar, input_formats_parsed, output_format_parsed)

    graph = to_iteration_graph(desugar, input_formats_parsed, output_format_parsed)
    problem = Problem(identifiable_assignment, input_formats_parsed, output_format_parsed)

    ir = generate_c_code(problem, graph, kernel_type).finalize()

    return ast_to_c(peephole(ir))
