from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class TensorLeaf:
    name: str
    instance: int

    def to_string(self):
        return f'{self.name}_{self.instance}'


def tensor_to_string(tensor: TensorLeaf):
    return f'{tensor.name}_{tensor.instance}'


def layer_pointer(tensor: Union[TensorLeaf, str], layer: int):
    return f'p_{tensor_to_string(tensor)}_{layer}'


def value_from_crd(tensor: TensorLeaf, layer: int):
    return f'i_{tensor_to_string(tensor)}_{layer}'


def tensor_to_pos(tensor: Union[TensorLeaf, str], layer: int):
    if isinstance(tensor, str):
        return f'{tensor}_{layer}_pos'
    else:
        return f'{tensor.name}_{layer}_pos'
