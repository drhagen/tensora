__all__ = ["to_iteration_graph"]

from dataclasses import replace
from functools import singledispatch
from itertools import count
from typing import Dict, Iterator

from ..format import Format
from ..iteration_graph import Lattice, LatticeLeaf
from ..iteration_graph import iteration_graph as graph
from ..iteration_graph.identifiable_expression import ast as id
from . import ast
from .collect_lattices import collect_lattices


def to_iteration_graph(
    assignment: ast.Assignment, formats: dict[str, Format]
) -> graph.IterationGraph:
    lattices = collect_lattices(assignment.expression, formats)

    tree = to_iteration_graph_expression(assignment.expression, lattices, formats, count(1))

    output_name = assignment.target.variable.name
    target_leaf = assignment.target.variable.to_tensor_leaf()
    variable = id.Tensor(target_leaf, assignment.target.indexes, formats[output_name].modes)
    for i, index_name in enumerate(assignment.target.indexes):
        tree = graph.IterationVariable(
            index_name, LatticeLeaf(variable, i), lattices[index_name], tree
        )

    return tree


@singledispatch
def to_iteration_graph_expression(
    expression: ast.DesugaredExpression,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    raise NotImplementedError(
        f"to_iteration_graph_expression not implemented for type {type(expression)}: {expression}"
    )


@to_iteration_graph_expression.register(ast.Integer)
def to_iteration_graph_integer(
    expression: ast.Integer,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    return graph.TerminalExpression(id.Integer(expression.value))


@to_iteration_graph_expression.register(ast.Float)
def to_iteration_graph_float(
    expression: ast.Float,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    return graph.TerminalExpression(id.Float(expression.value))


@to_iteration_graph_expression.register(ast.Scalar)
def to_iteration_graph_scalar(
    expression: ast.Scalar,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    target_leaf = expression.variable.to_tensor_leaf()
    return graph.TerminalExpression(id.Scalar(target_leaf))


@to_iteration_graph_expression.register(ast.Tensor)
def to_iteration_graph_tensor(
    expression: ast.Tensor,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    target_leaf = expression.variable.to_tensor_leaf()
    variable = id.Tensor(target_leaf, expression.indexes, formats[expression.variable.name].modes)
    return graph.TerminalExpression(variable)


@to_iteration_graph_expression.register(ast.Add)
def to_iteration_graph_add(
    expression: ast.Add,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    left = to_iteration_graph_expression(expression.left, lattices, formats, ids)
    right = to_iteration_graph_expression(expression.right, lattices, formats, ids)

    match (left, right):
        case (graph.TerminalExpression(), graph.TerminalExpression()):
            return graph.TerminalExpression(id.Add(left.expression, right.expression))
        case (graph.Add(name=name, terms=left_terms), graph.Add(terms=right_terms)):
            return graph.Add(name, left_terms + right_terms)
        case (graph.Add(name=name, terms=left_terms), _):
            return graph.Add(name, [*left_terms, right])
        case (_, graph.Add(name=name, terms=right_terms)):
            return graph.Add(name, [left, *right_terms])
        case (_, _):
            return graph.Add(f"sum{next(ids)}", [left, right])


@to_iteration_graph_expression.register(ast.Multiply)
def to_iteration_graph_multiply(
    expression: ast.Multiply,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    left = to_iteration_graph_expression(expression.left, lattices, formats, ids)
    right = to_iteration_graph_expression(expression.right, lattices, formats, ids)

    match (left, right):
        case (graph.TerminalExpression(), graph.TerminalExpression()):
            return graph.TerminalExpression(id.Multiply(left.expression, right.expression))
        case (
            graph.IterationVariable(next=graph.TerminalExpression() as next),
            graph.TerminalExpression(),
        ):
            # It is unclear if this is actually more efficient, but it is cleaner code
            return replace(
                left, next=graph.TerminalExpression(id.Multiply(next.expression, right.expression))
            )
        case (
            graph.TerminalExpression(),
            graph.IterationVariable(next=graph.TerminalExpression() as next),
        ):
            # It is unclear if this is actually more efficient, but it is cleaner code
            return replace(
                right, next=graph.TerminalExpression(id.Multiply(left.expression, next.expression))
            )
        case (_, _):
            raise NotImplementedError(
                "Multiplication of two iteration variables is not implemented"
            )


@to_iteration_graph_expression.register(ast.Contract)
def to_iteration_graph_contract(
    expression: ast.Contract,
    lattices: Dict[str, Lattice],
    formats: dict[str, Format],
    ids: Iterator[int],
):
    return graph.IterationVariable(
        expression.index,
        None,
        lattices[expression.index],
        to_iteration_graph_expression(expression.expression, lattices, formats, ids),
    )
