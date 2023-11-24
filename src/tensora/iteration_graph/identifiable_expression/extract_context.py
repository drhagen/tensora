from __future__ import annotations

__all__ = ["extract_context"]

from dataclasses import dataclass
from functools import singledispatch

from ...format import Mode
from . import ast
from .tensor_layer import TensorLayer


@dataclass(frozen=True, slots=True)
class Context:
    is_dense: bool
    sparse_leaves: list[TensorLayer]
    dense_leaves: list[TensorLayer]
    indexes: frozenset[str] = frozenset()

    def add(self, other: Context) -> Context:
        return Context(
            self.is_dense or other.is_dense,
            self.sparse_leaves + other.sparse_leaves,
            self.dense_leaves + other.dense_leaves,
            self.indexes | other.indexes,
        )

    def multiply(self, other: Context) -> Context:
        return Context(
            self.is_dense and other.is_dense,
            self.sparse_leaves + other.sparse_leaves,
            self.dense_leaves + other.dense_leaves,
            self.indexes | other.indexes,
        )


@singledispatch
def extract_context(self: ast.Expression, index: str) -> Context:
    raise NotImplementedError(f"extract_context not implemented for {type(self)}: {self}")


@extract_context.register(ast.Literal)
@extract_context.register(ast.Scalar)
def extract_context_scalar(self: ast.Literal, index: str) -> Context:
    if self == ast.Integer(0) or self == ast.Float(0.0):
        return Context(False, [], [])
    else:
        return Context(True, [], [])


@extract_context.register(ast.Tensor)
def extract_context_tensor(self: ast.Tensor, index: str) -> Context:
    try:
        maybe_layer = self.indexes.index(index)
    except ValueError:
        return Context(True, [], [])

    if self.modes[maybe_layer] == Mode.dense:
        return Context(True, [], [TensorLayer(self, maybe_layer)])
    else:
        return Context(False, [TensorLayer(self, maybe_layer)], [])


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
