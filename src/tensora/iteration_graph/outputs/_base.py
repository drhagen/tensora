from __future__ import annotations

__all__ = ["Output"]

from abc import abstractmethod

from ...ir import SourceBuilder
from ...ir.ast import Expression
from ...kernel_type import KernelType
from ..identifiable_expression import TensorLayer


class Output:
    __slots__ = ()

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
