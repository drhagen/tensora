__all__ = ["best_algorithm"]

from returns.result import Failure, Result, Success

from ..format import Format
from ..iteration_graph.iteration_graph import IterationGraph
from . import ast
from ._exceptions import DiagonalAccessError, NoKernelFoundError
from ._to_iteration_graphs import to_iteration_graphs


def best_algorithm(
    assignment: ast.Assignment, formats: dict[str, Format]
) -> Result[IterationGraph, DiagonalAccessError | NoKernelFoundError]:
    try:
        match next(to_iteration_graphs(assignment, formats), None):
            case None:
                return Failure(NoKernelFoundError())
            case graph:
                return Success(graph)
    except DiagonalAccessError as e:
        return Failure(e)
