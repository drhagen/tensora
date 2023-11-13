__all__ = ["Id"]

from dataclasses import dataclass


@dataclass(frozen=True)
class Id:
    name: str
    instance: int

    def to_tensor_leaf(self):
        from ..iteration_graph.identifiable_expression import TensorLeaf

        return TensorLeaf(self.name, self.instance)
