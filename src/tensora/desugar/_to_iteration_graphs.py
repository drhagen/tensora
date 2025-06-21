__all__ = ["to_iteration_graphs"]

from collections import defaultdict
from dataclasses import replace
from functools import reduce, singledispatch
from itertools import chain, count, permutations, product
from typing import Iterator

from ..format import Format, Mode
from ..iteration_graph import iteration_graph as ig
from ..iteration_graph.identifiable_expression import TensorLayer
from ..iteration_graph.identifiable_expression import ast as id
from . import ast


def legal_iteration_orders(format: Format) -> Iterator[list[int]]:
    """Legal iteration orders for layers of a tensor of a given format.

    Zero is the first layer, not the first index. For example, a ds format has
    one legal iteration order (i.e. [0, 1]) and a d1s0 format has a different
    legal iteration order (i.e. [1, 0])."""
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

    for group_order in product(*(permutations(group) for group in reorderable_groups)):
        yield list(chain(*group_order))


@singledispatch
def to_iteration_graphs_expression(
    self: ast.Expression,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    raise NotImplementedError(
        f"to_iteration_graphs_expression not implemented for {type(self)}: {self}"
    )


@to_iteration_graphs_expression.register(ast.Integer)
def to_iteration_graphs_integer(
    self: ast.Integer,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield ig.TerminalNode(id.Integer(self.value))


@to_iteration_graphs_expression.register(ast.Float)
def to_iteration_graphs_float(
    self: ast.Float,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield ig.TerminalNode(id.Float(self.value))


@to_iteration_graphs_expression.register(ast.Tensor)
def to_iteration_graphs_tensor(
    self: ast.Tensor,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    from ._exceptions import DiagonalAccessError

    format = formats[self.name]
    index_variables = tuple(self.indexes[i_index] for i_index in format.ordering)
    modes = formats[self.name].modes

    if len(set(index_variables)) != len(index_variables):
        raise DiagonalAccessError(self)

    for index_order in legal_iteration_orders(format):
        graph = ig.TerminalNode(
            id.Tensor(f"{self.id}_{self.name}", self.name, index_variables, modes)
        )
        # Build iteration order bottom up
        for i_index in reversed(index_order):
            index_variable = index_variables[i_index]
            graph = ig.IterationNode(
                index_variable,
                None,
                next=graph,
            )
        yield graph


def simplify_add(graph: ig.SumNode) -> ig.IterationGraph:
    """Simplify an Add by combining terms with the same index variable.

    An Add node can be simplified by combining all its `TerminalExpression`s into a single
    expression and by combinbing all its `IterationVariable`s with the same index variable into a
    single loop.
    """
    # graph.terms are guaranteed to be IterationVariable or TerminalExpression

    terminal_nodes: list[ig.TerminalNode] = []
    iteration_nodes: defaultdict[str, list[ig.IterationNode]] = defaultdict(list)
    for term in graph.terms:
        match term:
            case ig.TerminalNode():
                terminal_nodes.append(term)
            case ig.IterationNode():
                iteration_nodes[term.index_variable].append(term)

    combined_nodes = []

    if len(terminal_nodes) > 0:
        expression = reduce(id.Add, [term.expression for term in terminal_nodes])
        combined_nodes.append(ig.TerminalNode(expression))

    for terms in iteration_nodes.values():
        # Zip the iteration nodes as far as they will go
        head = terms[0]

        next_terms = []
        for term in terms:
            match term.next:
                case ig.SumNode():
                    # Merge sum nodes
                    next_terms.extend(term.next.terms)
                case _:
                    next_terms.append(term.next)

        combined_nodes.append(replace(head, next=simplify_add(ig.SumNode(graph.name, next_terms))))

    match combined_nodes:
        case [single_term]:
            return single_term
        case _:
            return ig.SumNode(graph.name, combined_nodes)


@to_iteration_graphs_expression.register(ast.Add)
def to_iteration_graphs_add(
    self: ast.Add,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    name = f"sum_{next(counter)}"
    for left in to_iteration_graphs_expression(self.left, formats, counter):
        for right in to_iteration_graphs_expression(self.right, formats, counter):
            # Always simplify Add within Add
            match (left, right):
                case (ig.SumNode(), ig.SumNode()):
                    graph = ig.SumNode(name, [*left.terms, *right.terms])
                case (ig.SumNode(), _):
                    graph = ig.SumNode(name, [*left.terms, right])
                case (_, ig.SumNode()):
                    graph = ig.SumNode(name, [left, *right.terms])
                case (_, _):
                    graph = ig.SumNode(name, [left, right])

            yield simplify_add(graph)


def merge_multiply(
    left: ig.IterationGraph, right: ig.IterationGraph
) -> Iterator[ig.IterationGraph]:
    match (left, right):
        case (ig.TerminalNode(), ig.TerminalNode()):
            yield ig.TerminalNode(id.Multiply(left.expression, right.expression))
        case (ig.IterationNode(), ig.TerminalNode()):
            for tail in merge_multiply(left.next, right):
                yield replace(left, next=tail)
        case (ig.TerminalNode(), ig.IterationNode()):
            for tail in merge_multiply(left, right.next):
                yield replace(right, next=tail)
        case (ig.IterationNode(), ig.IterationNode()):
            if left.index_variable == right.index_variable:
                for tail in merge_multiply(left.next, right.next):
                    yield replace(left, next=tail)
            else:
                if left.index_variable not in right.next.later_indexes():
                    for tail in merge_multiply(left.next, right):
                        yield replace(left, next=tail)

                if right.index_variable not in left.next.later_indexes():
                    for tail in merge_multiply(left, right.next):
                        yield replace(right, next=tail)
        case (ig.SumNode(), _):
            for terms in product(*[merge_multiply(term, right) for term in left.terms]):
                yield ig.SumNode(left.name, list(terms))
        case (_, ig.SumNode()):
            for terms in product(*[merge_multiply(left, term) for term in right.terms]):
                yield ig.SumNode(right.name, list(terms))


@to_iteration_graphs_expression.register(ast.Multiply)
def to_iteration_graphs_multiply(
    self: ast.Multiply,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    for left in to_iteration_graphs_expression(self.left, formats, counter):
        for right in to_iteration_graphs_expression(self.right, formats, counter):
            yield from merge_multiply(left, right)


@to_iteration_graphs_expression.register(ast.Contract)
def to_iteration_graphs_contract(
    self: ast.Contract,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[ig.IterationGraph]:
    yield from to_iteration_graphs_expression(self.expression, formats, counter)


def merge_assignment(
    target: ig.IterationGraph, expression: ig.IterationGraph, output_layers: dict[str, TensorLayer]
) -> Iterator[ig.IterationGraph]:
    match (target, expression):
        case (ig.TerminalNode(), _):
            yield expression
        case (ig.IterationNode(), ig.TerminalNode()):
            output_leaf = output_layers[target.index_variable]
            for tail in merge_assignment(target.next, expression, output_layers):
                yield replace(target, output=output_leaf, next=tail)
        case (ig.IterationNode(), ig.IterationNode()):
            if target.index_variable == expression.index_variable:
                output_leaf = output_layers[target.index_variable]
                for tail in merge_assignment(target.next, expression.next, output_layers):
                    yield replace(target, output=output_leaf, next=tail)
            else:
                if target.index_variable not in expression.next.later_indexes():
                    output_leaf = output_layers[target.index_variable]
                    for tail in merge_assignment(target.next, expression, output_layers):
                        yield replace(target, output=output_leaf, next=tail)

                if expression.index_variable not in target.next.later_indexes():
                    for tail in merge_assignment(target, expression.next, output_layers):
                        yield replace(expression, next=tail)
        case (ig.IterationNode(), ig.SumNode(name=name, terms=terms)):
            for merged_terms in product(
                *(merge_assignment(target, term, output_layers) for term in terms)
            ):
                yield simplify_add(ig.SumNode(name, list(merged_terms)))


def to_iteration_graphs(
    assignment: ast.Assignment, formats: dict[str, Format]
) -> Iterator[ig.IterationGraph]:
    output_format = formats[assignment.target.name]
    output_layers = {
        assignment.target.indexes[i_dimension]: TensorLayer(
            id.Tensor(
                f"{assignment.target.id}_{assignment.target.name}",
                assignment.target.name,
                tuple(assignment.target.indexes[i] for i in output_format.ordering),
                output_format.modes,
            ),
            i_layer,
        )
        for i_layer, i_dimension in enumerate(output_format.ordering)
    }

    for target_graph in to_iteration_graphs_expression(assignment.target, formats, []):
        for expression_graph in to_iteration_graphs_expression(
            assignment.expression, formats, count(1)
        ):
            for graph in merge_assignment(target_graph, expression_graph, output_layers):
                yield graph
