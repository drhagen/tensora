from dataclasses import dataclass


@dataclass(frozen=True)
class TensorLeaf:
    name: str
    instance: int

    def to_string(self):
        return f"{self.name}_{self.instance}"
