__all__ =["assignment_to_c_code"]

from typing import Dict
from tensora.codegen.ast_to_c import ast_to_c
from tensora.desugar.to_identifiable import to_identifiable
from tensora.desugar.desugar_expression import desugar_assignment
from tensora.desugar.to_iteration_graph import to_iteration_graph
from tensora.expression.parser import parse_assignment

from tensora.format.parser import parse_format
from tensora.ir.peephole import peephole
from tensora.iteration_graph.iteration_graph_to_c_code import KernelType, generate_c_code
from tensora.iteration_graph.problem import Problem


def assignment_to_c_code(string: str, input_formats: Dict[str, str], output_format: str, kernel_type: KernelType) -> str:
    assignment = parse_assignment(string).unwrap()
    input_formats_parsed = {name: parse_format(format).unwrap() for name, format in input_formats.items()}
    output_format_parsed = parse_format(output_format).unwrap()

    desugar = desugar_assignment(assignment)

    identifiable_assignment = to_identifiable(desugar, input_formats_parsed, output_format_parsed)

    graph = to_iteration_graph(desugar, input_formats_parsed, output_format_parsed)
    problem = Problem(identifiable_assignment, input_formats_parsed, output_format_parsed)

    ir = generate_c_code(problem, graph, kernel_type).finalize()

    return ast_to_c(peephole(ir))
