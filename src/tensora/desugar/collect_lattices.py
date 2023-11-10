__all__ = ["collect_lattices"]

from functools import singledispatch
from tensora.format.format import Format

from tensora.iteration_graph.merge_lattice.merge_lattice import LatticeConjuction, LatticeDisjunction, LatticeLeaf, Lattice
from . import ast
from ..iteration_graph.identifiable_expression import ast as id


@singledispatch
def collect_lattices(expression: ast.DesugaredExpression, formats: dict[str, Format]) -> dict[str, Lattice]:
    raise NotImplementedError(f"collect_lattices not implemented for type {type(expression)}: {expression}")


@collect_lattices.register(ast.Literal)
def collect_lattices_literal(expression: ast.Literal, formats: dict[str, Format]):
    return {}


@collect_lattices.register(ast.Scalar)
def collect_lattices_scalar(expression: ast.Scalar, formats: dict[str, Format]):
    return {}


@collect_lattices.register(ast.Tensor)
def collect_lattices_tensor(expression: ast.Tensor, formats: dict[str, Format]):
    target_leaf = expression.variable.to_tensor_leaf()
    variable = id.Tensor(target_leaf, expression.indexes, formats[expression.variable.name].modes)
    return {index: LatticeLeaf(variable, i) for i, index in enumerate(expression.indexes)}


@collect_lattices.register(ast.Add)
def collect_lattices_add(expression: ast.Add, formats: dict[str, Format]):
    lattice = collect_lattices(expression.left, formats)
    for index, lattice_i in collect_lattices(expression.right, formats).items():
        if index in lattice:
            lattice[index] = LatticeConjuction(lattice[index], lattice_i)
        else:
            lattice[index] = lattice_i
    return lattice


@collect_lattices.register(ast.Multiply)
def collect_lattices_multiply(expression: ast.Multiply, formats: dict[str, Format]):
    lattice = collect_lattices(expression.left, formats)
    for index, lattice_i in collect_lattices(expression.right, formats).items():
        if index in lattice:
            lattice[index] = LatticeDisjunction(lattice[index], lattice_i)
        else:
            lattice[index] = lattice_i
    return lattice


@collect_lattices.register(ast.Contract)
def collect_lattices_contract(expression: ast.Contract, formats: dict[str, Format]):
    return collect_lattices(expression.expression, formats)
