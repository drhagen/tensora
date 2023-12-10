__all__ = ["to_ir"]

from functools import singledispatch

from ...ir import ast as ir
from .ast import Add, Expression, Float, Integer, Multiply, Tensor


@singledispatch
def to_ir(self: Expression) -> ir.Expression:
    raise NotImplementedError(f"to_ir not implemented for {type(self)}: {self}")


@to_ir.register(Integer)
def to_ir_integer(self: Integer):
    # This is sensible as long as we only support floating point values and don't support division. If either of those
    # ceases to be true, this will need to be updated.
    return ir.IntegerLiteral(self.value)


@to_ir.register(Float)
def to_ir_float(self: Float):
    return ir.FloatLiteral(self.value)


@to_ir.register(Tensor)
def to_ir_tensor(self: Tensor):
    from .._names import previous_layer_pointer, vals_name

    return vals_name(self.name).idx(previous_layer_pointer(self.id, self.order))


@to_ir.register(Add)
def to_ir_add(self: Add):
    return ir.Add(to_ir(self.left), to_ir(self.right))


@to_ir.register(Multiply)
def to_ir_multiply(self: Multiply):
    return ir.Multiply(to_ir(self.left), to_ir(self.right))
