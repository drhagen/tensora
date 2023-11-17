__all__ = ["downstream_indexes"]

from functools import singledispatch

from . import iteration_graph as ig


@singledispatch
def downstream_indexes(self: ig.IterationGraph) -> set[str]:
    return self.indexes


@downstream_indexes.register(ig.TerminalExpression)
def downstream_indexes_terminal_expression(self: ig.TerminalExpression) -> set[str]:
    return set()


@downstream_indexes.register(ig.IterationVariable)
def downstream_indexes_iteration_variable(self: ig.IterationVariable) -> set[str]:
    return downstream_indexes(self.next) | {self.index_variable}


@downstream_indexes.register(ig.Add)
def downstream_indexes_add(self: ig.Add) -> set[str]:
    return set().union(*(downstream_indexes(term) for term in self.terms))
