__all__ = ["to_identifiable"]

from functools import singledispatch

from ..format import Format
from ..iteration_graph.identifiable_expression import ast as id
from . import ast as desugar


@singledispatch
def to_identifiable(self: desugar.Variable, formats: dict[str, Format]) -> id.Variable:
    raise NotImplementedError(f"to_identifiable not implemented for {type(self)}: {self}")


@to_identifiable.register(desugar.Scalar)
def to_identifiable_scalar(self: desugar.Scalar, formats: dict[str, Format]):
    return id.Scalar(id.TensorLeaf(self.name, self.id))


@to_identifiable.register(desugar.Tensor)
def to_identifiable_tensor(self: desugar.Tensor, formats: dict[str, Format]):
    format = formats[self.name]
    return id.Tensor(
        id.TensorLeaf(self.name, self.id),
        tuple(self.indexes[i_index] for i_index in format.ordering),
        format.modes,
    )
