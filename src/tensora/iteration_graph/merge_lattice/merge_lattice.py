from __future__ import annotations

__all__ = ['Lattice', 'LatticeLeaf', 'LatticeConjuction', 'LatticeDisjunction', 'IterationMode']

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, AbstractSet, List

from tensora import Mode
from tensora.iteration_graph.identifiable_expression import TensorLeaf
from tensora.iteration_graph.identifiable_expression.ast import Tensor
from tensora.stable_frozen_set import StableFrozenSet


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
    tensor: Tensor
    layer: int

    @property
    def mode(self):
        return self.tensor.modes[self.layer]

    def layer_pointer(self):
        return f'p_{self.tensor.variable.to_string()}_{self.layer}'

    def previous_layer_pointer(self):
        if self.layer == 0:
            return '0'
        else:
            return LatticeLeaf(self.tensor, self.layer - 1).layer_pointer()

    def value_from_crd(self):
        return f'i_{self.tensor.variable.to_string()}_{self.layer}'

    def pos_name(self):
        return f'{self.tensor.variable.name}_{self.layer}_pos'

    def crd_name(self):
        return f'{self.tensor.variable.name}_{self.layer}_crd'

    def vals_name(self):
        return f'{self.tensor.variable.name}_vals'

    def pos_capacity_name(self):
        return f'{self.tensor.variable.name}_{self.layer}_pos_capacity'

    def crd_capacity_name(self):
        return f'{self.tensor.variable.name}_{self.layer}_crd_capacity'

    def vals_capacity_name(self):
        return f'{self.tensor.variable.name}_vals_capacity'

    def dimension_name(self):
        return f'{self.tensor.indexes[self.layer]}_dim'

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
            return StableFrozenSet(self.tensor)
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
