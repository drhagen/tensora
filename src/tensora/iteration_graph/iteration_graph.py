from __future__ import annotations

__all__ = ["IterationGraph", "Add", "IterationVariable", "TerminalExpression"]

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
from .identifiable_expression.ast import Expression


class IterationGraph:
    @abstractmethod
    def extract_context(self, index: str) -> Context:
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

    def extract_context(self, index: str) -> Context:
        return extract_context(self.expression, index)

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
    output: TensorLayer | None
    next: IterationGraph

    def __post_init__(self):
        self.context: Context
        object.__setattr__(self, "context", self.extract_context(self.index_variable))

    def extract_context(self, index: str) -> Context:
        return self.next.extract_context(index)

    def exhaust_tensor(self, tensor: TensorLeaf) -> IterationGraph | None:
        new_next = self.next.exhaust_tensor(tensor)
        if new_next is not None:
            return replace(self, next=new_next)
        else:
            return None

    def is_dense(self) -> bool:
        return self.context.is_dense or self.output is not None and self.output.mode == Mode.dense

    def compressed_dimensions(self) -> StableFrozenSet[TensorLeaf]:
        return StableFrozenSet(*(leaf.tensor.variable for leaf in self.context.sparse_leaves))

    def sparse_leaves(self) -> list[TensorLayer]:
        return [TensorLayer(leaf.tensor, leaf.layer) for leaf in self.context.sparse_leaves]

    def dense_leaves(self) -> list[TensorLayer]:
        return [TensorLayer(leaf.tensor, leaf.layer) for leaf in self.context.dense_leaves]

    def all_dense(self) -> bool:
        return self.context.is_dense and self.next.all_dense()

    def is_dense_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.dense

    def is_sparse_output(self) -> bool:
        return self.output is not None and self.output.mode == Mode.compressed


@dataclass(frozen=True)
class Add(IterationGraph):
    name: str
    terms: list[IterationGraph]

    def extract_context(self, index: str) -> Context:
        context = Context(False, [], [])
        for term in self.terms:
            context = context.add(term.extract_context(index))
        return context

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
