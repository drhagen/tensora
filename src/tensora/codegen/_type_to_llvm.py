__all__ = [
    "llvm_integer_type",
    "llvm_float_type",
    "llvm_boolean_type",
    "llvm_mode_type",
    "llvm_size_type",
    "type_to_llvm",
    "attribute_indexes",
]

from functools import singledispatch

import llvmlite.ir as llvm

from ..ir.types import Array, FixedArray, Float, Integer, Mode, Pointer, Tensor, Type

llvm_integer_type = llvm.IntType(32)
llvm_float_type = llvm.DoubleType()
llvm_boolean_type = llvm.IntType(1)
llvm_mode_type = llvm.IntType(8)
llvm_size_type = llvm.IntType(64)


@singledispatch
def type_to_llvm(self: Type) -> llvm.Type:
    raise NotImplementedError(f"type_to_llvm not implemented for {type(self)}: {self}")


@type_to_llvm.register(Integer)
def type_to_llvm_integer(self: Integer) -> llvm.Type:
    return llvm_integer_type


@type_to_llvm.register(Float)
def type_to_llvm_float(self: Float) -> llvm.Type:
    return llvm_float_type


tensor_attribute_indexes = {
    "dimensions": 1,
    "indices": 5,
    "vals": 6,
}


@type_to_llvm.register(Tensor)
def type_to_llvm_tensor(self: Tensor) -> llvm.Type:
    return llvm.LiteralStructType(
        [
            llvm_integer_type,  # order
            llvm.PointerType(llvm_integer_type),  # dimensions
            llvm_integer_type,  # csize
            llvm.PointerType(llvm_integer_type),  # mode_ordering
            llvm_mode_type,  # mode_types
            llvm.PointerType(llvm.PointerType(llvm.PointerType(llvm_integer_type))),  # indices
            llvm.PointerType(llvm_float_type),  # vals
            llvm_integer_type,  # vals_size
        ]
    )


@type_to_llvm.register(Mode)
def type_to_llvm_mode(self: Mode) -> llvm.Type:
    return llvm_mode_type


@type_to_llvm.register(Pointer)
def type_to_llvm_pointer(self: Pointer) -> llvm.Type:
    return llvm.PointerType(type_to_llvm(self.target))


@type_to_llvm.register(Array)
def type_to_llvm_array(self: Array) -> llvm.Type:
    return llvm.PointerType(type_to_llvm(self.element))


@type_to_llvm.register(FixedArray)
def type_to_llvm_fixed_array(self: FixedArray) -> llvm.Type:
    return llvm.ArrayType(type_to_llvm(self.element), self.n)


# LLVM index for all attributes.
# If we ever reuse an attribute name, and it goes to a different index, the whole IR will have to
# be redesigned.
attribute_indexes = tensor_attribute_indexes
