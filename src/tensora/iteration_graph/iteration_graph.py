from __future__ import annotations

__all__ = ["IterationGraph", "Add", "IterationVariable", "TerminalExpression"]

from abc import abstractmethod
from dataclasses import dataclass, replace

from ..format import Mode
from .identifiable_expression import TensorLeaf, exhaust_tensor
from .identifiable_expression.ast import Expression
from .merge_lattice import Lattice, LatticeLeaf


class IterationGraph:
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
class Add(IterationGraph):
    name: str
    terms: list[IterationGraph]

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        new_terms = []
        for term in self.terms:
            new_term = term.exhaust_tensor(tensor)
            if new_term is not None:
                new_terms.append(new_term)

        if len(new_terms) == 0:
            return None
        else:
            return replace(self, terms=new_terms)

    def all_dense(self) -> bool:
        return True

    def is_sparse_output(self) -> bool:
        return False


@dataclass(frozen=True)
class IterationVariable(IterationGraph):
    index_variable: str
    output: LatticeLeaf | None
    lattice: Lattice
    next: IterationGraph

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
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

    def all_dense(self) -> bool:
        return self.lattice.is_dense() and self.next.all_dense()

    def is_sparse_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.compressed


@dataclass(frozen=True)
class TerminalExpression(IterationGraph):
    expression: Expression

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
