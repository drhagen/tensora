from __future__ import annotations

__all__ = ["Mode", "Format"]

from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    # Manually map these to the entries in .compile.taco_type_header.taco_mode_t
    dense = (0, "d")
    compressed = (1, "s")

    def __init__(self, c_int: int, character: "str"):
        self.c_int = c_int
        self.character = character

    @staticmethod
    def from_c_int(value: int) -> Mode:
        for member in Mode:
            if member.value[0] == value:
                return member
        raise ValueError(f"No member of Mode has the integer value {value}")

    def __repr__(self) -> str:
        return f"Mode.{self.name}"


@dataclass(frozen=True, slots=True)
class Format:
    modes: tuple[Mode, ...]
    ordering: tuple[int, ...]

    def __post_init__(self):
        from ._exceptions import InvalidModeOrderingError

        if set(self.ordering) != set(range(len(self.modes))):
            raise InvalidModeOrderingError(self.modes, self.ordering)

    @property
    def order(self):
        return len(self.modes)

    def deparse(self):
        if self.ordering == tuple(range(self.order)):
            return "".join(mode.character for mode in self.modes)
        else:
            return "".join(
                mode.character + str(ordering)
                for mode, ordering in zip(self.modes, self.ordering, strict=True)
            )
