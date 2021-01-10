from dataclasses import dataclass
from typing import Dict

from tensora import Format
from tensora.iteration_graph.identifiable_expression import Assignment
from tensora.iteration_graph.identifiable_expression.index_dimension import index_dimension
from tensora.iteration_graph.identifiable_expression.index_dimensions import index_dimensions


@dataclass(frozen=True)
class Problem:
    assignment: Assignment
    input_formats: Dict[str, Format]
    output_format: Format

    def formats(self):
        return {self.assignment.target.name: self.output_format, **self.input_formats}

    def index_dimension(self, index_variable: str):
        return index_dimension(self.assignment, index_variable)

    def index_dimensions(self):
        return index_dimensions(self.assignment)
