__all__ = ["DiagonalAccessError"]

from dataclasses import dataclass

from . import ast


@dataclass(frozen=True, slots=True)
class DiagonalAccessError(Exception):
    tensor: ast.Tensor

    def __str__(self) -> str:
        return (
            f"Diagonal access to a tensor (i.e. repeating the same index within a tensor) is not "
            f"currently supported: {self.tensor.variable.name}({', '.join(self.tensor.indexes)}"
        )
