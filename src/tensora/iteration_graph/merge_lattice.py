from __future__ import annotations

__all__ = ["Lattice", "LatticeLeaf", "LatticeConjuction", "LatticeDisjunction"]

from abc import abstractmethod
from dataclasses import dataclass

from ..format import Mode
from ..ir.ast import Expression, Variable
from ..stable_set import StableFrozenSet
from .identifiable_expression import TensorLeaf
from .identifiable_expression import ast as ie_ast
from .names import (
    crd_capacity_name,
    crd_name,
    layer_pointer,
    pos_capacity_name,
    pos_name,
    previous_layer_pointer,
    sparse_end_name,
    vals_capacity_name,
    vals_name,
    value_from_crd,
)


class Lattice:
    @abstractmethod
    def is_dense(self):
        pass

    @abstractmethod
    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        pass

    @abstractmethod
    def exhaust_tensor(self, tensor: TensorLeaf) -> Lattice | None:
        pass

    @abstractmethod
    def sparse_leaves(self) -> list[LatticeLeaf]:
        pass

    @abstractmethod
    def dense_leaves(self) -> list[LatticeLeaf]:
        pass


@dataclass(frozen=True)
class LatticeLeaf(Lattice):
    tensor: ie_ast.Variable
    layer: int

    @property
    def mode(self):
        return self.tensor.modes[self.layer]

    def layer_pointer(self) -> Variable:
        return layer_pointer(self.tensor.variable, self.layer)

    def sparse_end_name(self) -> Variable:
        return sparse_end_name(self.tensor.variable, self.layer)

    def previous_layer_pointer(self) -> Expression:
        return previous_layer_pointer(self.tensor.variable, self.layer)

    def value_from_crd(self) -> Variable:
        return value_from_crd(self.tensor.variable, self.layer)

    def pos_name(self) -> Variable:
        return pos_name(self.tensor.variable.name, self.layer)

    def crd_name(self) -> Variable:
        return crd_name(self.tensor.variable.name, self.layer)

    def vals_name(self) -> Variable:
        return vals_name(self.tensor.variable.name)

    def pos_capacity_name(self) -> Variable:
        return pos_capacity_name(self.tensor.variable.name, self.layer)

    def crd_capacity_name(self) -> Variable:
        return crd_capacity_name(self.tensor.variable.name, self.layer)

    def vals_capacity_name(self) -> Variable:
        return vals_capacity_name(self.tensor.variable.name)

    def is_dense(self) -> bool:
        return self.mode == Mode.dense

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        if self.mode == Mode.compressed:
            return StableFrozenSet(self.tensor.variable)
        else:
            return StableFrozenSet()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Lattice | None:
        if self.tensor.variable == tensor:
            # This will only happen when mode == compressed
            return None
        else:
            return self

    def sparse_leaves(self) -> list[LatticeLeaf]:
        if self.mode == Mode.dense:
            return []
        elif self.mode == Mode.compressed:
            return [self]

    def dense_leaves(self) -> list[LatticeLeaf]:
        if self.mode == Mode.dense:
            return [self]
        elif self.mode == Mode.compressed:
            return []


@dataclass(frozen=True)
class LatticeConjuction(Lattice):
    left: Lattice
    right: Lattice

    def is_dense(self):
        return self.left.is_dense() or self.right.is_dense()

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return self.left.compressed_dimensions() | self.right.compressed_dimensions()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Lattice | None:
        left_exhausted = self.left.exhaust_tensor(tensor)
        right_exhausted = self.right.exhaust_tensor(tensor)
        if left_exhausted is self.left and right_exhausted is self.right:
            # Short circuit when there are no changes
            return self
        elif left_exhausted is None:
            # Covers the case where both are exhausted
            return right_exhausted
        elif right_exhausted is None:
            return left_exhausted
        else:
            return LatticeConjuction(left_exhausted, right_exhausted)

    def sparse_leaves(self) -> list[LatticeLeaf]:
        return self.left.sparse_leaves() + self.right.sparse_leaves()

    def dense_leaves(self) -> list[LatticeLeaf]:
        return self.left.dense_leaves() + self.right.dense_leaves()


@dataclass(frozen=True)
class LatticeDisjunction(Lattice):
    left: Lattice
    right: Lattice

    def is_dense(self):
        return self.left.is_dense() and self.right.is_dense()

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return self.left.compressed_dimensions() | self.right.compressed_dimensions()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Lattice | None:
        left_exhausted = self.left.exhaust_tensor(tensor)
        right_exhausted = self.right.exhaust_tensor(tensor)
        if left_exhausted is self.left and right_exhausted is self.right:
            # Short circuit when there are no changes
            return self
        elif left_exhausted is None or right_exhausted is None:
            return None
        else:
            return LatticeDisjunction(left_exhausted, right_exhausted)

    def sparse_leaves(self) -> list[LatticeLeaf]:
        return self.left.sparse_leaves() + self.right.sparse_leaves()

    def dense_leaves(self) -> list[LatticeLeaf]:
        return self.left.dense_leaves() + self.right.dense_leaves()
