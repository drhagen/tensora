__all__ = ["TensorLayer"]

from dataclasses import dataclass

from ...ir import ast as ir
from .._names import (
    crd_capacity_name,
    crd_name,
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
from . import ast as id


@dataclass(frozen=True, slots=True)
class TensorLayer:
    tensor: id.Tensor
    layer: int

    @property
    def mode(self):
        return self.tensor.modes[self.layer]

    def pos_name(self) -> ir.Variable:
        return pos_name(self.tensor.name, self.layer)

    def crd_name(self) -> ir.Variable:
        return crd_name(self.tensor.name, self.layer)

    def vals_name(self) -> ir.Variable:
        return vals_name(self.tensor.name)

    def pos_capacity_name(self) -> ir.Variable:
        return pos_capacity_name(self.tensor.name, self.layer)

    def crd_capacity_name(self) -> ir.Variable:
        return crd_capacity_name(self.tensor.name, self.layer)

    def vals_capacity_name(self) -> ir.Variable:
        return vals_capacity_name(self.tensor.name)

    def layer_pointer(self) -> ir.Variable:
        return layer_pointer(self.tensor.id, self.layer)

    def previous_layer_pointer(self) -> ir.Expression:
        return previous_layer_pointer(self.tensor.id, self.layer)

    def sparse_end_name(self) -> ir.Variable:
        return sparse_end_name(self.tensor.id, self.layer)

    def layer_begin_name(self) -> ir.Variable:
        return layer_begin_name(self.tensor.id, self.layer)

    def value_from_crd(self) -> ir.Variable:
        return value_from_crd(self.tensor.id, self.layer)
