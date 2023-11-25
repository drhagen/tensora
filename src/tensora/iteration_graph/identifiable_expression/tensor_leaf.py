__all__ = ["TensorLeaf"]

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorLeaf:
    name: str
    instance: int
