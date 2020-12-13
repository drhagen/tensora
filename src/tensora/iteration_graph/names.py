from typing import Union, List

from .identifiable_expression import TensorLeaf


def dimension_name(index_variable: str):
    return f'{index_variable}_dim'


def pos_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_pos'


def crd_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_crd'


def vals_name(tensor: str):
    return f'{tensor}_vals'


def crd_capacity_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_crd_capacity'


def pos_capacity_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_pos_capacity'


def vals_capacity_name(tensor: str):
    return f'{tensor}_vals_capacity'


def tensor_to_string(tensor: Union[TensorLeaf, str]):
    if isinstance(tensor, str):
        return tensor
    else:
        return f'{tensor.name}_{tensor.instance}'


def layer_pointer(tensor: Union[TensorLeaf, str], layer: int):
    return f'p_{tensor_to_string(tensor)}_{layer}'


def value_from_crd(tensor: Union[TensorLeaf, str], layer: int):
    return f'i_{tensor_to_string(tensor)}_{layer}'


def tensor_to_pos(tensor: Union[TensorLeaf, str], layer: int):
    if isinstance(tensor, str):
        return f'{tensor}_{layer}_pos'
    else:
        return f'{tensor.name}_{layer}_pos'


def hash_table_name(tensor: Union[TensorLeaf, str], starting_layer: int):
    return f'hash_table_{tensor_to_string(tensor)}_{starting_layer}'


def bucket_name(tensor: Union[TensorLeaf, str], indexes: List[int]):
    return f'bucket_{tensor_to_string(tensor)}{"".join(f"_{x}" for x in indexes)}'
