from __future__ import annotations

__all__ = [
    "dimension_name",
    "pos_name",
    "crd_name",
    "vals_name",
    "pos_capacity_name",
    "crd_capacity_name",
    "vals_capacity_name",
    "layer_pointer",
    "previous_layer_pointer",
    "sparse_end_name",
    "value_from_crd",
]


from ..ir.ast import Expression, IntegerLiteral, Variable


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


def layer_pointer(reference: str, layer: int) -> Variable:
    return Variable(f"p_{reference}_{layer}")


def previous_layer_pointer(reference: str, layer: int) -> Expression:
    if layer == 0:
        return IntegerLiteral(0)
    else:
        return layer_pointer(reference, layer - 1)


def sparse_end_name(reference: str, layer: int) -> Variable:
    return Variable(f"p_{reference}_{layer}_end")


def layer_begin_name(reference: str, layer: int) -> Variable:
    return Variable(f"p_{reference}_{layer}_begin")


def value_from_crd(reference: str, layer: int) -> Variable:
    return Variable(f"i_{reference}_{layer}")
