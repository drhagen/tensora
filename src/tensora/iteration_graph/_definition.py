__all__ = ["Definition", "TensorDimension"]

from dataclasses import dataclass

from ..format import Format
from .identifiable_expression.ast import Tensor


@dataclass(frozen=True, slots=True)
class TensorDimension:
    name: str
    dimension: int


@dataclass(frozen=True, slots=True)
class Definition:
    output_variable: Tensor
    formats: dict[str, Format]
    indexes: dict[str, TensorDimension]
