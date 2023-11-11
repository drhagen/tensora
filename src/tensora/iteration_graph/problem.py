__all__ = ["Problem"]

from dataclasses import dataclass
from typing import Dict

from ..format import Format
from .identifiable_expression import Assignment, index_dimension, index_dimensions


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
