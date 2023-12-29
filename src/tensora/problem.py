__all__ = [
    "Problem",
    "make_problem",
    "IncorrectDimensionsError",
    "UndefinedReferenceError",
    "UnusedFormatError",
]

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from .expression.ast import Assignment
from .format import Format, Mode


@dataclass(frozen=True, slots=True)
class IncorrectDimensionsError(Exception):
    name: str
    actual: int
    expected: int
    assignment: Assignment

    def __str__(self):
        return (
            f"Expected each reference in an assignment to have a number of indexes matching the "
            f"order of the corresponding format, but variable {self.name} referenced in "
            f"{self.assignment} indexes has order {self.actual} while its format has order {self.expected}"
        )


@dataclass(frozen=True, slots=True)
class UndefinedReferenceError(Exception):
    name: str
    assignment: Assignment
    formats: list[str]

    def __str__(self):
        return (
            f"Excepted each reference in an assignment to have a corresponding format, "
            f"but variable {self.name} referenced in {self.assignment} was not found among the "
            f"given formats {self.formats}"
        )


@dataclass(frozen=True, slots=True)
class UnusedFormatError(Exception):
    name: str
    assignment: Assignment

    def __str__(self):
        return (
            f"Expected each format to be referenced in the assignment, "
            f"but format {self.name} was not referenced in {self.assignment}"
        )


@dataclass(frozen=True, slots=True)
class Problem:
    assignment: Assignment
    formats: dict[str, Format]

    def __post_init__(self):
        # This intentionally allows for names in formats that are not referenced in the assignment.
        # The CLI and porcelain API will not allow this, but this is just as valid as defining a
        # function with unused parameters.

        tensor_orders = self.assignment.variable_orders()
        for name, order in tensor_orders.items():
            if name not in self.formats:
                raise UndefinedReferenceError(name, self.assignment, list(self.formats.keys()))
            elif order != self.formats[name].order:
                raise IncorrectDimensionsError(
                    name, self.formats[name].order, order, self.assignment
                )

    def __eq__(self, other: object):
        if isinstance(other, Problem):
            # Problems are only equal if the formats orders are equal
            return self.assignment == other.assignment and tuple(self.formats.items()) == tuple(
                other.formats.items()
            )
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.assignment, tuple(self.formats.items())))


def make_problem(
    assignment: Assignment, formats: dict[str, Format]
) -> Result[Problem, UnusedFormatError | UndefinedReferenceError | IncorrectDimensionsError]:
    """Create a Problem while filling in default formats.

    This does three things that the `Problem` constructor does not do:
    1. It reorders the formats to match the order the tensors appear in the assignment.
    2. It fills in any missing formats with all dense modes.
    3. It raises an exception if there are formats not referenced in the assignment.
    """

    tensor_orders = assignment.variable_orders()

    for name in formats.keys():
        if name not in tensor_orders:
            return Failure(UnusedFormatError(name, assignment))

    new_formats = {}
    for name, order in tensor_orders.items():
        if name not in formats:
            new_formats[name] = Format(tuple([Mode.dense] * order), tuple(range(order)))
        else:
            new_formats[name] = formats[name]

    try:
        problem = Problem(assignment, new_formats)
    except (UndefinedReferenceError, IncorrectDimensionsError) as error:
        return Failure(error)

    return Success(problem)
