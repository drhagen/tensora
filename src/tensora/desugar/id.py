from dataclasses import dataclass




@dataclass(frozen=True)
class Id:
    name: str
    instance: int

    def to_tensor_leaf(self):
        from tensora.iteration_graph.identifiable_expression.tensor_leaf import TensorLeaf
        return TensorLeaf(self.name, self.instance)