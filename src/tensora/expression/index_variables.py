__all__ = ["index_variables"]

from functools import singledispatch
from typing import List

from .ast import Add, Assignment, Literal, Multiply, Node, Subtract, Variable


@singledispatch
def index_variables(self: Node) -> List[str]:
    raise NotImplementedError()


@index_variables.register(Literal)
def index_variables_literal(self: Literal):
    return []


@index_variables.register(Variable)
def index_variables_variable(self: Variable):
    return self.indexes


@index_variables.register(Add)
def index_variables_add(self: Add):
    left_variables = index_variables(self.left)
    right_variables = index_variables(self.right)
    return left_variables + [
        variable for variable in right_variables if variable not in left_variables
    ]


@index_variables.register(Subtract)
def index_variables_subtract(self: Add):
    left_variables = index_variables(self.left)
    right_variables = index_variables(self.right)
    return left_variables + [
        variable for variable in right_variables if variable not in left_variables
    ]


@index_variables.register(Multiply)
def index_variables_multiply(self: Add):
    left_variables = index_variables(self.left)
    right_variables = index_variables(self.right)
    return left_variables + [
        variable for variable in right_variables if variable not in left_variables
    ]


@index_variables.register(Assignment)
def index_variables_assignment(self: Assignment):
    target_variables = index_variables(self.target)
    expression_variables = index_variables(self.expression)
    return target_variables + [
        variable for variable in expression_variables if variable not in target_variables
    ]
