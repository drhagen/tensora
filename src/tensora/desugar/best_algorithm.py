__all__ = ["best_algorithm"]

from ..format import Format
from ..iteration_graph.iteration_graph import IterationGraph
from . import ast
from .exceptions import NoKernelFoundError
from .to_iteration_graphs import to_iteration_graphs


def best_algorithm(
    assignment: ast.Assignment, formats: dict[str, Format | None]
) -> IterationGraph:
    match next(to_iteration_graphs(assignment, formats), None):
        case None:
            raise NoKernelFoundError()
        case graph:
            return graph
