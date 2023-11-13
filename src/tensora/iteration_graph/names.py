from ..ir.ast import Expression, IntegerLiteral, Variable
from .identifiable_expression import TensorLeaf


def dimension_name(index_variable: str) -> Variable:
    return Variable(f"{index_variable}_dim")


def pos_name(tensor: str, layer: int) -> Variable:
    return Variable(f"{tensor}_{layer}_pos")


def crd_name(tensor: str, layer: int) -> Variable:
    return Variable(f"{tensor}_{layer}_crd")


def vals_name(tensor: str) -> Variable:
    return Variable(f"{tensor}_vals")


def pos_capacity_name(tensor: str, layer: int) -> Variable:
    return Variable(f"{tensor}_{layer}_pos_capacity")


def crd_capacity_name(tensor: str, layer: int) -> Variable:
    return Variable(f"{tensor}_{layer}_crd_capacity")


def vals_capacity_name(tensor: str) -> Variable:
    return Variable(f"{tensor}_vals_capacity")


def tensor_to_string(tensor: TensorLeaf):
    return f"{tensor.name}_{tensor.instance}"


def layer_pointer(tensor: TensorLeaf, layer: int) -> Variable:
    return Variable(f"p_{tensor_to_string(tensor)}_{layer}")


def previous_layer_pointer(tensor: TensorLeaf, layer: int) -> Expression:
    if layer == 0:
        return IntegerLiteral(0)
    else:
        return layer_pointer(tensor, layer - 1)


def sparse_end_name(tensor: TensorLeaf, layer: int) -> Variable:
    return Variable(f"p_{tensor_to_string(tensor)}_{layer}_end")


def layer_begin_name(tensor: TensorLeaf, layer: int) -> Variable:
    return Variable(f"p_{tensor_to_string(tensor)}_{layer}_begin")


def value_from_crd(tensor: TensorLeaf, layer: int) -> Variable:
    return Variable(f"i_{tensor_to_string(tensor)}_{layer}")
