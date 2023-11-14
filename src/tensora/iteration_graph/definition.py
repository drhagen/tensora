__all__ = ["Definition", "TensorLayer"]

from dataclasses import dataclass

from ..format import Format
from .identifiable_expression.ast import Variable


@dataclass(frozen=True, slots=True)
class TensorLayer:
    name: str
    dimension: int


@dataclass(frozen=True, slots=True)
class Definition:
    output_variable: Variable
    formats: dict[str, Format]
    indexes: dict[str, TensorLayer]
