__all__ = ["InvalidModeOrderingError"]

from dataclasses import dataclass

from ._format import Mode


@dataclass(frozen=True, slots=True)
class InvalidModeOrderingError(Exception):
    modes: tuple[Mode, ...]
    ordering: tuple[int, ...]

    def __str__(self):
        return (
            f"Expected ordering to have be some order of integers 0 until length of modes, "
            f"but got modes={self.modes} and ordering={self.ordering}"
        )
