__all__ = ["to_iteration_graphs"]

from dataclasses import replace
from functools import reduce, singledispatch
from itertools import count, product
from typing import Iterator

from ..format import Format, Mode
from ..iteration_graph import Lattice, LatticeLeaf, downstream_indexes
from ..iteration_graph import iteration_graph as ig
from ..iteration_graph.identifiable_expression import ast as id
from . import ast
from .collect_lattices import collect_lattices


def legal_iteration_orders(format: Format) -> Iterator[list[int]]:
    """Legal iteration orders for indexes of a tensor of a given format.

    Zero is the first index, not the first layer. For example, a ds format has
    one legal iteration order (i.e. [1, 0]) and a d1s0 format has a different
    legal iteration order (i.e. [0, 1])."""
    reorderable_groups = []
    restart = True
    for i, mode in enumerate(format.modes):
        match mode:
            case Mode.dense:
                if restart:
                    reorderable_groups.append([i])
                else:
                    reorderable_groups[-1].append(i)
                restart = False
            case Mode.compressed:
                reorderable_groups.append([i])
                restart = True

    for layer_order in product(*reorderable_groups):
        yield [format.ordering[i_layer] for i_layer in layer_order]


@singledispatch
def to_iteration_graphs_expression(
    self: ast.DesugaredExpression,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    raise NotImplementedError(
        f"to_iteration_graphs_expression not implemented for type {type(self)}: {self}"
    )


@to_iteration_graphs_expression.register(ast.Integer)
def to_iteration_graphs_integer(
    self: ast.Integer,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield ig.TerminalExpression(id.Integer(self.value))


@to_iteration_graphs_expression.register(ast.Float)
def to_iteration_graphs_float(
    self: ast.Float,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield ig.TerminalExpression(id.Float(self.value))


@to_iteration_graphs_expression.register(ast.Scalar)
def to_iteration_graphs_scalar(
    self: ast.Scalar,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield ig.TerminalExpression(id.Scalar(self.variable.to_tensor_leaf()))


@to_iteration_graphs_expression.register(ast.Tensor)
def to_iteration_graphs_tensor(
    self: ast.Tensor,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    index_variables = self.indexes
    modes = formats[self.variable.name].modes

    for index_order in legal_iteration_orders(formats[self.variable.name]):
        graph = ig.TerminalExpression(
            id.Tensor(self.variable.to_tensor_leaf(), tuple(index_variables), modes)
        )
        # Build iteration order bottom up
        for i_index in reversed(index_order):
            index_variable = index_variables[i_index]
            graph = ig.IterationVariable(
                index_variable,
                None,
                lattice=lattices[index_variable],
                next=graph,
            )
        yield graph


def simplify_add(graph: ig.Add) -> ig.IterationGraph:
    """Simplify an Add by combining terms with the same index variable.

    An Add node can be simplified if all its terms are `IterationVariable`s with
    the same index variable. In this case, the IterationVariable can be pulled
    above the Add node.
    """
    # graph.terms are guaranteed to be IterationVariable or TerminalExpression

    # This could yield all the intermediate graphs, but the last one might
    # always be the most efficient.

    if all(isinstance(term, ig.IterationVariable) for term in graph.terms):
        unique_terms = {term.index_variable for term in graph.terms}
        if len(unique_terms) == 1:
            head = graph.terms[0]
            return replace(
                head, next=simplify_add(ig.Add(graph.name, [term.next for term in graph.terms]))
            )
        else:
            return graph
    elif all(isinstance(term, ig.TerminalExpression) for term in graph.terms):
        expression = reduce(id.Add, [term.expression for term in graph.terms])
        return ig.TerminalExpression(expression)
    else:
        return graph


@to_iteration_graphs_expression.register(ast.Add)
def to_iteration_graphs_add(
    self: ast.Add,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    name = f"sum_{next(counter)}"
    for left in to_iteration_graphs_expression(self.left, formats, lattices, counter):
        for right in to_iteration_graphs_expression(self.right, formats, lattices, counter):
            # Always simplify Add within Add
            match (left, right):
                case (ig.Add(), ig.Add()):
                    graph = ig.Add(name, [*left.terms, *right.terms])
                case (ig.Add(), _):
                    graph = ig.Add(name, [*left.terms, right])
                case (_, ig.Add()):
                    graph = ig.Add(name, [left, *right.terms])
                case (_, _):
                    graph = ig.Add(name, [left, right])

            while True:
                yield simplify_add(graph)


def merge_multiply(
    left: ig.IterationGraph, right: ig.IterationGraph
) -> Iterator[ig.IterationGraph]:
    match (left, right):
        case (ig.TerminalExpression(), ig.TerminalExpression()):
            yield ig.TerminalExpression(id.Multiply(left.expression, right.expression))
        case (ig.IterationVariable(), ig.TerminalExpression()):
            for tail in merge_multiply(left.next, right):
                yield replace(left, next=tail)
        case (ig.TerminalExpression(), ig.IterationVariable()):
            for tail in merge_multiply(left, right.next):
                yield replace(right, next=tail)
        case (ig.IterationVariable(), ig.IterationVariable()):
            if left.index_variable == right.index_variable:
                for tail in merge_multiply(left.next, right.next):
                    yield replace(left, next=tail)
            else:
                if left.index_variable not in downstream_indexes(right.next):
                    for tail in merge_multiply(left.next, right):
                        yield replace(left, next=tail)

                if right.index_variable not in downstream_indexes(left.next):
                    for tail in merge_multiply(left, right.next):
                        yield replace(right, next=tail)
        case (ig.Add(), _):
            for terms in product(*[merge_multiply(term, right) for term in left.terms]):
                yield ig.Add(left.name, list(terms))
        case (_, ig.Add()):
            for terms in product(*[merge_multiply(left, term) for term in right.terms]):
                yield ig.Add(right.name, list(terms))


@to_iteration_graphs_expression.register(ast.Multiply)
def to_iteration_graphs_multiply(
    self: ast.Multiply,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    for left in to_iteration_graphs_expression(self.left, formats, lattices, counter):
        for right in to_iteration_graphs_expression(self.right, formats, lattices, counter):
            yield from merge_multiply(left, right)


@to_iteration_graphs_expression.register(ast.Contract)
def to_iteration_graphs_contract(
    self: ast.Contract,
    formats: dict[str, Format],
    lattices: dict[str, Lattice],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield from to_iteration_graphs_expression(self.expression, formats, lattices, counter)


def merge_assignment(
    target: ig.IterationGraph, expression: ig.IterationGraph, output_layers: dict[str, LatticeLeaf]
) -> Iterator[ig.IterationGraph]:
    match (target, expression):
        case (ig.TerminalExpression(), _):
            yield expression
        case (ig.IterationVariable(), ig.TerminalExpression()):
            output_leaf = output_layers[target.index_variable]
            for tail in merge_assignment(target.next, expression, output_layers):
                yield replace(target, output=output_leaf, next=tail)
        case (ig.IterationVariable(), ig.IterationVariable()):
            if target.index_variable == expression.index_variable:
                output_leaf = output_layers[target.index_variable]
                for tail in merge_assignment(target.next, expression.next, output_layers):
                    yield replace(target, output=output_leaf, next=tail)
            else:
                if target.index_variable not in downstream_indexes(expression.next):
                    output_leaf = output_layers[target.index_variable]
                    for tail in merge_assignment(target.next, expression, output_layers):
                        yield replace(target, output=output_leaf, next=tail)

                if expression.index_variable not in downstream_indexes(target.next):
                    for tail in merge_assignment(target, expression.next, output_layers):
                        yield replace(expression, next=tail)
        case (ig.IterationVariable(), ig.Add()):
            # No iteration variables allowed downstream of an Add
            pass


def to_iteration_graphs(
    assignment: ast.Assignment, formats: dict[str, Format]
) -> Iterator[ig.IterationGraph]:
    lattices = collect_lattices(assignment.expression, formats)
    output_layers = {
        index_variable: LatticeLeaf(
            id.Tensor(
                assignment.target.variable.to_tensor_leaf(),
                tuple(assignment.target.indexes),
                formats[assignment.target.variable.name].modes,
            ),
            i,
        )
        for i, index_variable in enumerate(assignment.target.indexes)
    }

    for expression_graph in to_iteration_graphs_expression(
        assignment.expression, formats, lattices, count(1)
    ):
        for target_graph in to_iteration_graphs_expression(
            assignment.target, formats, lattices, []
        ):
            for graph in merge_assignment(target_graph, expression_graph, output_layers):
                yield graph
