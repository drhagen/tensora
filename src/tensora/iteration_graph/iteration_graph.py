from __future__ import annotations

__all__ = ["IterationGraph", "Sum", "IterationVariable", "TerminalExpression"]

from abc import abstractmethod
from dataclasses import dataclass, replace

from ..format import Mode
from ..stable_set import StableFrozenSet
from .identifiable_expression import (
    Context,
    TensorLayer,
    TensorLeaf,
    exhaust_tensor,
    extract_context,
)
from .identifiable_expression.ast import Expression, Integer


class IterationGraph:
    @abstractmethod
    def extract_context(self, index: str) -> Context:
        raise NotImplementedError()

    @abstractmethod
    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph:
        raise NotImplementedError()

    @abstractmethod
    def is_sparse_output(self) -> bool:
        # Needed by assembly to determine if next layer is guaranteed to advance position or not
        raise NotImplementedError()

    @abstractmethod
    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        # Needed when empty subgraphs simplify
        raise NotImplementedError()

    @abstractmethod
    def later_indexes(self) -> frozenset[str]:
        raise NotImplementedError()

    @abstractmethod
    def has_output(self) -> bool:
        raise NotImplementedError()


@dataclass(frozen=True)
class TerminalExpression(IterationGraph):
    expression: Expression

    def extract_context(self, index: str) -> Context:
        return extract_context(self.expression, index)

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph:
        return TerminalExpression(exhaust_tensor(self.expression, tensor))

    def is_sparse_output(self) -> bool:
        return False

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        # Needed when empty subgraphs simplify
        return StableFrozenSet()

    def later_indexes(self) -> frozenset[str]:
        return frozenset()

    def has_output(self) -> bool:
        return False


@dataclass(frozen=True)
class IterationVariable(IterationGraph):
    index_variable: str
    output: TensorLayer | None
    next: IterationGraph

    def __post_init__(self):
        self.context: Context
        object.__setattr__(self, "context", self.extract_context(self.index_variable))

    def extract_context(self, index: str) -> Context:
        next_context = self.next.extract_context(index)
        return replace(
            next_context,
            indexes=next_context.indexes | frozenset([self.index_variable]),
            has_output=next_context.has_output or self.output is not None,
            has_assemble=next_context.has_assemble or self.is_sparse_output(),
        )

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph:
        new_next = self.next.exhaust_tensor(tensor)

        return replace(self, next=new_next)

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return StableFrozenSet(*(leaf.tensor.variable for leaf in self.context.sparse_leaves))

    def sparse_leaves(self) -> list[TensorLayer]:
        return [TensorLayer(leaf.tensor, leaf.layer) for leaf in self.context.sparse_leaves]

    def dense_leaves(self) -> list[TensorLayer]:
        return [TensorLayer(leaf.tensor, leaf.layer) for leaf in self.context.dense_leaves]

    def is_sparse_input(self) -> bool:
        return self.context.is_sparse

    def is_dense_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.dense

    def is_sparse_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.compressed

    def later_indexes(self) -> frozenset[str]:
        return self.context.indexes

    def has_output(self) -> bool:
        return self.context.has_output

    def has_assemble(self) -> bool:
        return self.context.has_assemble


@dataclass(frozen=True)
class Sum(IterationGraph):
    name: str
    terms: list[IterationGraph]

    def extract_context(self, index: str) -> Context:
        context = Context(is_sparse=True)
        for term in self.terms:
            context = context.add(term.extract_context(index))
        return context

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph:
        new_terms = []
        for term in self.terms:
            new_term = term.exhaust_tensor(tensor)
            # TODO: Simplify empty terms
            new_terms.append(new_term)

        if len(new_terms) == 0:
            return TerminalExpression(Integer(0))
        elif len(new_terms) == 1:
            return new_terms[0]
        else:
            return replace(self, terms=new_terms)

    def is_sparse_output(self) -> bool:
        return False

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        # Needed when empty subgraphs simplify
        return StableFrozenSet()

    def later_indexes(self) -> frozenset[str]:
        return frozenset.union(*(term.later_indexes() for term in self.terms))

    def has_output(self) -> bool:
        return any(term.has_output() for term in self.terms)
