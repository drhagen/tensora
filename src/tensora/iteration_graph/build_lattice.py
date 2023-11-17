__all__ = ["build_lattice"]

from functools import singledispatch

from .identifiable_expression import ast
from .merge_lattice import Lattice, LatticeConjunction, LatticeDisjunction, LatticeLeaf


@singledispatch
def build_lattice(self: ast.Expression, index: str) -> Lattice | None:
    raise NotImplementedError(f"build_lattice not implemented for {type(self)}: {self}")


@build_lattice.register(ast.Literal)
@build_lattice.register(ast.Scalar)
def build_lattice_scalar(self: ast.Literal, index: str) -> Lattice | None:
    # TODO: Make this a scalar after refactoring the lattice data structure
    raise NotImplementedError(f"build_lattice not implemented for {type(self)}: {self}")


@build_lattice.register(ast.Tensor)
def build_lattice_tensor(self: ast.Tensor, index: str) -> Lattice | None:
    try:
        maybe_layer = self.indexes.index(index)
    except ValueError:
        return None

    return LatticeLeaf(self, maybe_layer)


@build_lattice.register(ast.Add)
def build_lattice_add(self: ast.Add, index: str) -> Lattice | None:
    left = build_lattice(self.left, index)
    right = build_lattice(self.right, index)
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return LatticeConjunction(left, right)


@build_lattice.register(ast.Multiply)
def build_lattice_multiply(self: ast.Multiply, index: str) -> Lattice | None:
    left = build_lattice(self.left, index)
    right = build_lattice(self.right, index)
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return LatticeDisjunction(left, right)
