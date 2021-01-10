from itertools import product, permutations, chain
from typing import Callable

from .iteration_graph import IterationVariable, TerminalExpression, IterationGraph
from .merge_lattice import Lattice
from ..format import Format, Mode
from ..expression.ast import Variable, Expression
from ..problem.problem import Problem


def generate_iteration_graphs_from_tensor(variable: Variable, format: Format):
    # Yield every legal iteration graph for this variable for the given format
    # Legal iteration graphs always iterate each dimension in order except adjacent dense dimensions, which may be
    # iterated in any order.

    # The index variables in mode order
    ordered_indexes = [variable.indexes[dimension] for dimension in format.ordering]

    # The dimensions that can be iterated over
    lattices = [Lattice(variable.name, i_layer, mode) for i_layer, mode in enumerate(format.modes)]

    # Indexes that can be permuted form a cluster. Dense modes while they exist will append to the cluster of previous
    # dense modes, otherwise the mode forms a new length-1 cluster.
    clusters = [[]]
    previous_mode_dense = True
    for index_variable, layer in zip(ordered_indexes, lattices):
        if layer.mode == Mode.dense:
            if previous_mode_dense:
                clusters[-1].append((index_variable, layer))
            else:
                clusters.append([(index_variable, layer)])
            previous_mode_dense = True
        elif layer.mode == Mode.compressed:
            clusters.append([(index_variable, layer)])
            previous_mode_dense = False

    for nested_list in product(*(permutations(cluster) for cluster in clusters)):
        flattened_list = list(chain.from_iterable(nested_list))
        graph = TerminalExpression(variable)
        for index_variable, layer in flattened_list:
            graph = IterationVariable(index_variable, layer, graph)

        yield graph


def conjunctive_merge(left: IterationGraph, right: IterationGraph,
                      operation: Callable[[Expression, Expression], Expression]):
    pass

def problem_to_iteration_graph(problem: Problem):
    tensors = []
    1
