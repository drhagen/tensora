from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import List, Optional

from .identifiable_expression.exhaust_tensors import exhaust_tensor
from .merge_lattice import Lattice, LatticeLeaf
from .identifiable_expression import Expression, TensorLeaf
from .. import Mode


class IterationGraph:
    @abstractmethod
    def exhaust_tensor(self, tensor: TensorLeaf):
        raise NotImplementedError()

    @abstractmethod
    def all_dense(self):
        raise NotImplementedError()

    @abstractmethod
    def is_sparse_output(self):
        # Needed by assembly to determine if next layer is guaranteed to advance position or not
        pass


@dataclass(frozen=True)
class Add(IterationGraph):
    name: str
    terms: List[IterationGraph]

    def exhaust_tensor(self, tensor: TensorLeaf):
        new_terms = []
        for term in self.terms:
            new_term = term.exhaust_tensor(tensor)
            if new_term is not None:
                new_terms.append(new_term)

        if len(new_terms) == 0:
            return None
        else:
            return replace(self, terms=new_terms)

    def all_dense(self):
        return True

    def is_sparse_output(self):
        return False


@dataclass(frozen=True)
class Multiply(IterationGraph):
    factors: List[IterationGraph]


@dataclass(frozen=True)
class Contract(IterationGraph):
    next: IterationGraph

    def exhaust_tensor(self, tensor: TensorLeaf):
        new_next = self.next.exhaust_tensor(tensor)
        if new_next is not None:
            return replace(self, next=new_next)
        else:
            return None

    def all_dense(self):
        return True

    def is_sparse_output(self):
        return False


@dataclass(frozen=True)
class IterationVariable(IterationGraph):
    index_variable: str
    output: Optional[LatticeLeaf]
    lattice: Lattice
    next: IterationGraph

    def exhaust_tensor(self, tensor: TensorLeaf):
        new_lattice = self.lattice.exhaust_tensor(tensor)
        if new_lattice is not None:
            new_next = self.next.exhaust_tensor(tensor)
            if new_next is not None:
                return replace(self, lattice=new_lattice, next=new_next)
            else:
                # I'm not sure if this is possible
                return None
        else:
            return None

    def all_dense(self):
        return self.lattice.is_dense() and self.next.all_dense()

    def is_sparse_output(self):
        return self.output is not None and self.output.mode == Mode.compressed


@dataclass(frozen=True)
class TerminalExpression(IterationGraph):
    expression: Expression

    def exhaust_tensor(self, tensor: TensorLeaf):
        return TerminalExpression(exhaust_tensor(self.expression, tensor))

    def all_dense(self):
        return True

    def is_sparse_output(self):
        return False
