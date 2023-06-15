__all__ = ["to_iteration_graph"]

from functools import singledispatch

from . import ast
from ..iteration_graph import iteration_graph as graph
from ..iteration_graph.identifiable_expression import ast as id

def to_iteration_graph(expression: ast.Assignment) -> graph.IterationGraph:
    pass


@singledispatch
def to_iteration_graph_expression(expression: ast.DesugaredExpression) -> graph.IterationGraph:
    raise NotImplementedError(f"to_iteration_graph_expression not implemented for type {type(expression)}: {expression}")


@to_iteration_graph_expression.register(ast.Integer)
def to_iteration_graph_integer(expression: ast.Integer):
    return graph.TerminalExpression(id.Integer(expression.value))


@to_iteration_graph_expression.register(ast.Float)
def to_iteration_graph_float(expression: ast.Float):
    return graph.TerminalExpression(id.Float(expression.value))


@to_iteration_graph_expression.register(ast.Variable)
def to_iteration_graph_float(expression: ast.Variable):
    return graph.IterationVariable(
        index_variable=expression.variable.name,
    )
