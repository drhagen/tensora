__all__ = ["DiagonalAccessError", "NoKernelFoundError"]

from dataclasses import dataclass

from . import ast


@dataclass(frozen=True, slots=True)
class DiagonalAccessError(Exception):
    tensor: ast.Tensor

    def __str__(self) -> str:
        return (
            f"Diagonal access to a tensor (i.e. repeating the same index within a tensor) is not "
            f"currently supported: {self.tensor.name}({', '.join(self.tensor.indexes)}"
        )


@dataclass(frozen=True, slots=True)
class NoKernelFoundError(Exception):
    def __str__(self) -> str:
        return (
            "Was unable to find a kernel to solve the given tensor expression. This is likely "
            "due to sparse tensors needing to be iterated in opposite orders."
        )
