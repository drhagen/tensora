from functools import singledispatch
from itertools import chain, permutations, product
from typing import Iterator

from ..format import Format, Mode
from ..iteration_graph import iteration_graph as ig
from ..iteration_graph.workspaces import Allocation, Workspace
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
def to_workspaces_expression(
    expression: ast.Expression, formats: dict[str, Format], counter: Iterator[int]
) -> Iterator[Workspace]:
    raise NotImplementedError(
        f"to_workspaces_expression not implemented for {type(expression)}: {expression}"
    )


@to_workspaces_expression.register
def to_workspaces_integer(
    self: ast.Integer, formats: dict[str, Format], counter: Iterator[int]
) -> Iterator[Workspace]:
    yield Workspace(
        allocation=Allocation(
            name=f"temp_{next(counter)}",
            indexes=[],
            modes=[],
        ),
        iteration=ig.TerminalNode(id.Integer(self.value)),
        dependencies=[],
    )


@to_workspaces_expression.register
def to_workspaces_float(
    self: ast.Float, formats: dict[str, Format], counter: Iterator[int]
) -> Iterator[Workspace]:
    yield Workspace(
        allocation=Allocation(
            name=f"temp_{next(counter)}",
            indexes=[],
            modes=[],
        ),
        iteration=ig.TerminalNode(id.Float(self.value)),
        dependencies=[],
    )


@to_workspaces_expression.register
def to_workspaces_tensor(
    self: ast.Tensor, formats: dict[str, Format], counter: Iterator[int]
) -> Iterator[Workspace]:
    format = formats[self.name]
    index_variables = tuple(self.indexes[i_index] for i_index in format.ordering)
    modes = formats[self.name].modes

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

        yield Workspace(
            allocation=Allocation(
                name=self.name,
                indexes=index_variables,
                modes=modes,
            ),
            iteration=graph,
            dependencies=[],
        )


@to_workspaces_expression.register
def to_workspaces_add(
    self: ast.Add,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[Workspace]:
    name = f"sum_{next(counter)}"
    for left in to_workspaces_expression(self.left, formats, counter):
        for right in to_workspaces_expression(self.right, formats, counter):
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


@to_workspaces_expression.register
def to_workspaces_multiply(
    self: ast.Multiply,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[Workspace]:
    for left in to_workspaces_expression(self.left, formats, counter):
        for right in to_workspaces_expression(self.right, formats, counter):
            yield from merge_multiply(left, right)


@to_workspaces_expression.register
def to_workspaces_contract(
    self: ast.Contract,
    formats: dict[str, Format],
    counter: Iterator[int],
) -> Iterator[Workspace]:
    yield from to_workspaces_expression(self.expression, formats, counter)


def to_workspaces(assignment: ast.Assignment, formats: dict[str, Format]) -> Iterator[Workspace]:
    pass
