from __future__ import annotations

__all__ = ["extract_context"]

from dataclasses import dataclass, field
from functools import singledispatch

from ...format import Mode
from . import ast
from ._tensor_layer import TensorLayer


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    is_sparse: bool
    sparse_leaves: list[TensorLayer] = field(default_factory=list)
    dense_leaves: list[TensorLayer] = field(default_factory=list)
    indexes: frozenset[str] = frozenset()
    has_output: bool = False
    has_assemble: bool = False

    def add(self, other: Context) -> Context:
        return Context(
            is_sparse=self.is_sparse and other.is_sparse,
            sparse_leaves=self.sparse_leaves + other.sparse_leaves,
            dense_leaves=self.dense_leaves + other.dense_leaves,
            indexes=self.indexes | other.indexes,
            has_output=self.has_output or other.has_output,
            has_assemble=self.has_assemble or other.has_assemble,
        )

    def multiply(self, other: Context) -> Context:
        return Context(
            is_sparse=self.is_sparse or other.is_sparse,
            sparse_leaves=self.sparse_leaves + other.sparse_leaves,
            dense_leaves=self.dense_leaves + other.dense_leaves,
            indexes=self.indexes | other.indexes,
            has_output=self.has_output or other.has_output,
            has_assemble=self.has_assemble or other.has_assemble,
        )


@singledispatch
def extract_context(self: ast.Expression, index: str) -> Context:
    raise NotImplementedError(f"extract_context not implemented for {type(self)}: {self}")


@extract_context.register(ast.Literal)
def extract_context_literal(self: ast.Literal, index: str) -> Context:
    if self == ast.Integer(0) or self == ast.Float(0.0):
        return Context(is_sparse=True)
    else:
        return Context(is_sparse=False)


@extract_context.register(ast.Tensor)
def extract_context_tensor(self: ast.Tensor, index: str) -> Context:
    try:
        maybe_layer = self.indexes.index(index)
    except ValueError:
        return Context(is_sparse=False)

    if self.modes[maybe_layer] == Mode.dense:
        return Context(is_sparse=False, dense_leaves=[TensorLayer(self, maybe_layer)])
    else:
        return Context(is_sparse=True, sparse_leaves=[TensorLayer(self, maybe_layer)])


@extract_context.register(ast.Add)
def extract_context_add(self: ast.Add, index: str) -> Context:
    left = extract_context(self.left, index)
    right = extract_context(self.right, index)
    return left.add(right)


@extract_context.register(ast.Multiply)
def extract_context_multiply(self: ast.Multiply, index: str) -> Context:
    left = extract_context(self.left, index)
    right = extract_context(self.right, index)
    return left.multiply(right)
