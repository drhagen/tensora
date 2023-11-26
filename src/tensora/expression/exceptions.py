__all__ = ["MutatingAssignmentError", "InconsistentVariableSizeError"]

from dataclasses import dataclass

from .ast import Assignment, Variable


@dataclass(frozen=True, slots=True)
class MutatingAssignmentError(Exception):
    assignment: Assignment

    def __str__(self):
        return (
            f"Expected assignment target to never appear on the right hand side, "
            f"but found {self.assignment.target.name} on both sides of {self.assignment}"
        )


@dataclass(frozen=True, slots=True)
class InconsistentVariableSizeError(Exception):
    assignment: Assignment
    first: Variable
    second: Variable

    def __str__(self):
        return (
            f"Expected each tensor in an assignment to be referenced with the same number of "
            f"indexes, but found parameter {self.first.name} referenced as {self.first} and then "
            f"as {self.second} in {self.assignment}"
        )
