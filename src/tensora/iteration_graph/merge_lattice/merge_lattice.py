from __future__ import annotations

__all__ = ["Lattice", "LatticeLeaf", "LatticeConjuction", "LatticeDisjunction", "IterationMode"]

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AbstractSet, List, Optional

from ...format import Mode
from ...ir.ast import Expression, Variable
from ...stable_set import StableFrozenSet
from ..identifiable_expression import TensorLeaf
from ..identifiable_expression import ast as ie_ast
from ..names import (
    crd_capacity_name,
    crd_name,
    dimension_name,
    layer_begin_name,
    layer_pointer,
    pos_capacity_name,
    pos_name,
    previous_layer_pointer,
    sparse_end_name,
    vals_capacity_name,
    vals_name,
    value_from_crd,
)


class IterationMode(Enum):
    dense_only = auto()
    sparse_only = auto()
    dense_and_sparse = auto()


class Lattice:
    @abstractmethod
    def is_dense(self):
        pass

    @abstractmethod
    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        pass

    @abstractmethod
    def exhaust_tensor(self, tensor: TensorLeaf) -> Optional[Lattice]:
        pass

    @abstractmethod
    def iteration_mode(self) -> IterationMode:
        pass

    @abstractmethod
    def sparse_tensors(self) -> AbstractSet[TensorLeaf]:
        pass

    @abstractmethod
    def leaves(self) -> List[LatticeLeaf]:
        pass

    @abstractmethod
    def sparse_leaves(self) -> List[LatticeLeaf]:
        pass

    @abstractmethod
    def dense_leaves(self) -> List[LatticeLeaf]:
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

    def layer_begin_name(self) -> Variable:
        return layer_begin_name(self.tensor.variable, self.layer)

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

    def dimension_name(self) -> Variable:
        return dimension_name(self.tensor.indexes[self.layer])

    def is_dense(self) -> bool:
        return self.mode == Mode.dense

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        if self.mode == Mode.compressed:
            return StableFrozenSet(self.tensor.variable)
        else:
            return StableFrozenSet()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Optional[Lattice]:
        if self.tensor.variable == tensor:
            # This will only happen when mode == compressed
            return None
        else:
            return self

    def iteration_mode(self):
        if self.mode == Mode.dense:
            return IterationMode.dense_only
        elif self.mode == Mode.compressed:
            return IterationMode.sparse_only
        else:
            raise NotImplementedError()

    def sparse_tensors(self) -> AbstractSet[TensorLeaf]:
        if self.mode == Mode.compressed:
            return StableFrozenSet(self.tensor.variable)
        else:
            return StableFrozenSet()

    def leaves(self):
        return [self]

    def sparse_leaves(self) -> List[LatticeLeaf]:
        if self.mode == Mode.dense:
            return []
        elif self.mode == Mode.compressed:
            return [self]

    def dense_leaves(self) -> List[LatticeLeaf]:
        if self.mode == Mode.dense:
            return [self]
        elif self.mode == Mode.compressed:
            return []

    def next_layer(self) -> Optional[LatticeLeaf]:
        next_layer = self.layer + 1
        if next_layer == len(self.tensor.modes):
            return None
        else:
            return LatticeLeaf(self.tensor, next_layer)


@dataclass(frozen=True)
class LatticeConjuction(Lattice):
    left: Lattice
    right: Lattice

    def is_dense(self):
        return self.left.is_dense() or self.right.is_dense()

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return self.left.compressed_dimensions() | self.right.compressed_dimensions()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Optional[Lattice]:
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

    def iteration_mode(self):
        left_iteration_mode = self.left.iteration_mode()
        right_iteration_mode = self.right.iteration_mode()
        if left_iteration_mode == right_iteration_mode:
            return left_iteration_mode
        else:
            return IterationMode.dense_and_sparse

    def sparse_tensors(self) -> AbstractSet[TensorLeaf]:
        return self.left.sparse_tensors() | self.right.sparse_tensors()

    def leaves(self) -> List[LatticeLeaf]:
        return self.left.leaves() + self.right.leaves()

    def sparse_leaves(self) -> List[LatticeLeaf]:
        return self.left.sparse_leaves() + self.right.sparse_leaves()

    def dense_leaves(self) -> List[LatticeLeaf]:
        return self.left.dense_leaves() + self.right.dense_leaves()


@dataclass(frozen=True)
class LatticeDisjunction(Lattice):
    left: Lattice
    right: Lattice

    def is_dense(self):
        return self.left.is_dense() and self.right.is_dense()

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return self.left.compressed_dimensions() | self.right.compressed_dimensions()

    def exhaust_tensor(self, tensor: TensorLeaf) -> Optional[Lattice]:
        left_exhausted = self.left.exhaust_tensor(tensor)
        right_exhausted = self.right.exhaust_tensor(tensor)
        if left_exhausted is self.left and right_exhausted is self.right:
            # Short circuit when there are no changes
            return self
        elif left_exhausted is None or right_exhausted is None:
            return None
        else:
            return LatticeDisjunction(left_exhausted, right_exhausted)

    def iteration_mode(self):
        left_iteration_mode = self.left.iteration_mode()
        right_iteration_mode = self.right.iteration_mode()
        if left_iteration_mode == right_iteration_mode:
            return left_iteration_mode
        else:
            return IterationMode.dense_and_sparse

    def sparse_tensors(self) -> AbstractSet[TensorLeaf]:
        return self.left.sparse_tensors() | self.right.sparse_tensors()

    def leaves(self) -> List[LatticeLeaf]:
        return self.left.leaves() + self.right.leaves()

    def sparse_leaves(self) -> List[LatticeLeaf]:
        return self.left.sparse_leaves() + self.right.sparse_leaves()

    def dense_leaves(self) -> List[LatticeLeaf]:
        return self.left.dense_leaves() + self.right.dense_leaves()
