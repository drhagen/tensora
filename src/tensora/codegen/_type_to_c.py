__all__ = ["type_to_c"]

from functools import singledispatch

from ..ir.types import Array, FixedArray, Float, Integer, Mode, Pointer, Tensor, Type


def space_variable(variable: str | None = None) -> str:
    if variable is None:
        return ""
    else:
        return f" {variable}"


@singledispatch
def type_to_c(self: Type, variable: str | None = None) -> str:
    raise NotImplementedError(f"type_to_c not implemented for {type(self)}: {self}")


@type_to_c.register(Integer)
def type_to_c_integer(self: Integer, variable: str | None = None) -> str:
    return "int32_t" + space_variable(variable)


@type_to_c.register(Float)
def type_to_c_float(self: Float, variable: str | None = None) -> str:
    return "double" + space_variable(variable)


@type_to_c.register(Tensor)
def type_to_c_tensor(self: Tensor, variable: str | None = None) -> str:
    return "taco_tensor_t" + space_variable(variable)


@type_to_c.register(Mode)
def type_to_c_mode(self: Mode, variable: str | None = None) -> str:
    return "taco_mode_t" + space_variable(variable)


@type_to_c.register(Pointer)
def type_to_c_pointer(self: Pointer, variable: str | None = None) -> str:
    return f"{type_to_c(self.target)}* restrict" + space_variable(variable)


@type_to_c.register(Array)
def type_to_c_array(self: Array, variable: str | None = None) -> str:
    return f"{type_to_c(self.element, variable)}[]"


@type_to_c.register(FixedArray)
def type_to_c_fixed_array(self: FixedArray, variable: str | None = None) -> str:
    return f"{type_to_c(self.element, variable)}[{self.n}]"
