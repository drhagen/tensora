__all__ = ['to_c_code']

from functools import singledispatch

from .ast import *


@singledispatch
def to_c_code(self: Expression) -> str:
    pass


@to_c_code.register(Integer)
def to_c_code_integer(self: Integer):
    # This is sensible as long as we only support floating point values and don't support division. If either of those
    # ceases to be true, this will need to be updated.
    return str(self.value)


@to_c_code.register(Float)
def to_c_code_float(self: Float):
    return str(self.value)


@to_c_code.register(Scalar)
def to_c_code_variable(self: Scalar):
    return self.variable.name


@to_c_code.register(Tensor)
def to_c_code_variable(self: Tensor):
    from ..names import layer_pointer
    return f'{self.variable.name}_vals[{layer_pointer(self.variable, len(self.indexes) - 1)}]'


@to_c_code.register(Add)
def to_c_code_add(self: Add):
    return f'({to_c_code(self.left)} + {to_c_code(self.right)})'


@to_c_code.register(Subtract)
def to_c_code_subtract(self: Subtract):
    return f'({to_c_code(self.left)} - {to_c_code(self.right)})'


@to_c_code.register(Multiply)
def to_c_code_multiply(self: Multiply):
    return f'({to_c_code(self.left)} * {to_c_code(self.right)})'
