__all__ = ["to_identifiable"]

from ..format import Format
from ..iteration_graph.identifiable_expression import ast as id
from . import ast as desugar


def to_identifiable(self: desugar.Tensor, formats: dict[str, Format]) -> id.Tensor:
    format = formats[self.name]
    return id.Tensor(
        f"{self.id}_{self.name}",
        self.name,
        tuple(self.indexes[i_index] for i_index in format.ordering),
        format.modes,
    )
