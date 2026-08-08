from __future__ import annotations

__all__ = ["Output"]

from abc import abstractmethod

from ...format import Mode
from ...ir import SourceBuilder
from ...ir.ast import Expression, Variable
from ...kernel_type import KernelType
from .._names import value_written_name
from ..identifiable_expression import TensorLayer
from ..identifiable_expression import ast as ie_ast


class Output:
    __slots__ = ()

    # Subclasses assemble this output tensor.
    output: ie_ast.Tensor

    def written_flags(self) -> list[Variable]:
        # Each compressed output layer stores a coordinate only if a value is written below it. The
        # terminal signals that by setting a per-layer "value written" flag. The flag name is a
        # pure function of the layer, and every compressed output layer sits above every terminal,
        # so the set of flags to raise is recovered from the output format rather than threaded
        # down through the iteration.
        return [
            value_written_name(self.output.id, layer)
            for layer, mode in enumerate(self.output.modes)
            if mode == Mode.compressed
        ]

    def has_active_flags(self) -> bool:
        # A contraction or sum subtree must run structurally (even in the assemble kernel) when it
        # gates a compressed output layer's flag. Since every compressed output layer sits above
        # every contraction, this reduces to whether the output has any compressed layer at all.
        return any(mode == Mode.compressed for mode in self.output.modes)

    @abstractmethod
    def write_assignment(
        self, right_hand_side: Expression, kernel_type: KernelType
    ) -> SourceBuilder:
        raise NotImplementedError()

    @abstractmethod
    def next_output(
        self, iteration_output: TensorLayer | None, kernel_type: KernelType
    ) -> tuple[Output, SourceBuilder, SourceBuilder]:
        raise NotImplementedError()
