from __future__ import annotations

__all__ = ["Workspace", "Allocation"]


from dataclasses import dataclass

from ...format import Mode
from ..iteration_graph import IterationGraph


@dataclass(frozen=True, slots=True)
class Workspace:
    allocation: Allocation
    iteration: IterationGraph
    dependencies: list[Workspace]


@dataclass(frozen=True, slots=True)
class Allocation:
    name: str
    indexes: list[str]
    modes: list[Mode]
