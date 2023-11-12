from functools import singledispatch

from .ast import Expression, Literal, Scalar, Tensor, Add, Subtract, Multiply
from .tensor_leaf import TensorLeaf


@singledispatch
def exhaust_tensor(self, tensor: TensorLeaf) -> Expression:
    pass


@exhaust_tensor.register(Literal)
def exhaust_tensor_literal(self: Literal, tensor: TensorLeaf):
    return self


@exhaust_tensor.register(Scalar)
def exhaust_tensor_scalar(self: Scalar, tensor: TensorLeaf):
    return self


@exhaust_tensor.register(Tensor)
def exhaust_tensor_tensor(self: Tensor, tensor: TensorLeaf):
    if self.variable == tensor:
        return None
    else:
        return self


@exhaust_tensor.register(Add)
def exhaust_tensor_add(self: Add, tensor: TensorLeaf):
    left_exhausted = exhaust_tensor(self.left, tensor)
    right_exhausted = exhaust_tensor(self.right, tensor)
    if left_exhausted is self.left and right_exhausted is self.right:
        # Short circuit when there are no changes
        return self
    elif left_exhausted is None:
        # Covers the case where both are exhausted
        return right_exhausted
    elif right_exhausted is None:
        return left_exhausted
    else:
        return Add(left_exhausted, right_exhausted)


@exhaust_tensor.register(Subtract)
def exhaust_tensor_subtract(self: Subtract, tensor: TensorLeaf):
    left_exhausted = exhaust_tensor(self.left, tensor)
    right_exhausted = exhaust_tensor(self.right, tensor)
    if left_exhausted is self.left and right_exhausted is self.right:
        # Short circuit when there are no changes
        return self
    elif left_exhausted is None:
        # Covers the case where both are exhausted
        return right_exhausted
    elif right_exhausted is None:
        return left_exhausted
    else:
        return Subtract(left_exhausted, right_exhausted)


@exhaust_tensor.register(Multiply)
def exhaust_tensor_multiply(self: Multiply, tensor: TensorLeaf):
    left_exhausted = exhaust_tensor(self.left, tensor)
    right_exhausted = exhaust_tensor(self.right, tensor)
    if left_exhausted is self.left and right_exhausted is self.right:
        # Short circuit when there are no changes
        return self
    elif left_exhausted is None or right_exhausted is None:
        return None
    else:
        return Multiply(left_exhausted, right_exhausted)
