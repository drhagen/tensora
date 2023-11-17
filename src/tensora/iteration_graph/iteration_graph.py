from __future__ import annotations

__all__ = ["IterationGraph", "Add", "IterationVariable", "TerminalExpression"]

from abc import abstractmethod
from dataclasses import dataclass, replace

from ..format import Mode
from .build_lattice import build_lattice
from .identifiable_expression import TensorLeaf, exhaust_tensor
from .identifiable_expression.ast import Expression
from .merge_lattice import Lattice, LatticeConjunction, LatticeLeaf


class IterationGraph:
    @abstractmethod
    def build_lattice(self, index: str) -> Lattice | None:
        raise NotImplementedError()

    @abstractmethod
    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        raise NotImplementedError()

    @abstractmethod
    def all_dense(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_sparse_output(self) -> bool:
        # Needed by assembly to determine if next layer is guaranteed to advance position or not
        raise NotImplementedError()


@dataclass(frozen=True)
class TerminalExpression(IterationGraph):
    expression: Expression

    def build_lattice(self, index: str) -> Lattice | None:
        return build_lattice(self.expression, index)

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        expression = exhaust_tensor(self.expression, tensor)
        if expression is None:
            return None
        else:
            return TerminalExpression(expression)

    def all_dense(self) -> bool:
        return True

    def is_sparse_output(self) -> bool:
        return False


@dataclass(frozen=True)
class IterationVariable(IterationGraph):
    index_variable: str
    output: LatticeLeaf | None
    next: IterationGraph

    def __post_init__(self):
        self.lattice: Lattice
        object.__setattr__(self, "lattice", self.build_lattice(self.index_variable))

    def build_lattice(self, index: str) -> Lattice | None:
        return self.next.build_lattice(index)

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        new_next = self.next.exhaust_tensor(tensor)
        if new_next is not None:
            return replace(self, next=new_next)
        else:
            return None

    def all_dense(self) -> bool:
        return self.lattice.is_dense() and self.next.all_dense()

    def is_sparse_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.compressed


@dataclass(frozen=True)
class Add(IterationGraph):
    name: str
    terms: list[IterationGraph]

    def build_lattice(self, index: str) -> Lattice | None:
        lattice = None
        for term in self.terms:
            lattice_i = term.build_lattice(index)
            if lattice_i is None:
                pass
            elif lattice is None:
                lattice = lattice_i
            else:
                lattice = LatticeConjunction(lattice, lattice_i)
        return lattice

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        new_terms = []
        for term in self.terms:
            new_term = term.exhaust_tensor(tensor)
            if new_term is not None:
                new_terms.append(new_term)

        if len(new_terms) == 0:
            return None
        elif len(new_terms) == 1:
            return new_terms[0]
        else:
            return replace(self, terms=new_terms)

    def all_dense(self) -> bool:
        return True

    def is_sparse_output(self) -> bool:
        return False
