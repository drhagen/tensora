from __future__ import annotations

__all__ = ["Output"]

from abc import abstractmethod
from dataclasses import replace

from ...ir import SourceBuilder
from ...ir.ast import Expression, Variable
from ...kernel_type import KernelType
from ..identifiable_expression import TensorLayer


class Output:
    __slots__ = ()

    # Subclasses store the tuple of "value written" flags gating the enclosing sparse output
    # layers. A layer whose crd append is conditional on whether anything was written below it
    # threads its flag down to the terminal, which sets it when a contribution branch executes.
    gating_flags: tuple[Variable, ...]

    def has_active_flags(self) -> bool:
        return len(self.gating_flags) > 0

    def with_flag(self, flag: Variable) -> Output:
        return replace(self, gating_flags=(*self.gating_flags, flag))

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
