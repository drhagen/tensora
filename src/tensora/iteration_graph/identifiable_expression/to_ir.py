__all__ = ["to_ir"]

from functools import singledispatch

from ...ir import ast as ir
from .ast import Add, Expression, Float, Integer, Multiply, Scalar, Subtract, Tensor


@singledispatch
def to_ir(self: Expression) -> ir.Expression:
    raise NotImplementedError(f"to_ir not implemented for {type(self)}: {self}")


@to_ir.register(Integer)
def to_c_code_integer(self: Integer):
    # This is sensible as long as we only support floating point values and don't support division. If either of those
    # ceases to be true, this will need to be updated.
    return ir.IntegerLiteral(self.value)


@to_ir.register(Float)
def to_c_code_float(self: Float):
    return ir.FloatLiteral(self.value)


@to_ir.register(Scalar)
def to_c_code_scalar(self: Scalar):
    return ir.Variable(self.variable.name)


@to_ir.register(Tensor)
def to_c_code_tensor(self: Tensor):
    from ..names import previous_layer_pointer, vals_name

    return vals_name(self.variable.name).idx(previous_layer_pointer(self.variable, self.order))


@to_ir.register(Add)
def to_c_code_add(self: Add):
    return ir.Add(to_ir(self.left), to_ir(self.right))


@to_ir.register(Subtract)
def to_c_code_subtract(self: Subtract):
    return ir.Subtract(to_ir(self.left), to_ir(self.right))


@to_ir.register(Multiply)
def to_c_code_multiply(self: Multiply):
    return ir.Multiply(to_ir(self.left), to_ir(self.right))
