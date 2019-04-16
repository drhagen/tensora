__all__ = ['Mode', 'Format']

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Mode(Enum):
    # Manually map these to the entries in .taco_compile.taco_type_header.taco_mode_t
    dense = (0, 'd')
    compressed = (1, 's')

    def __init__(self, c_int: int, character: 'str'):
        self.c_int = c_int
        self.character = character

    @staticmethod
    def from_c_int(value: int) -> 'Mode':
        for member in Mode:
            if member.value[0] == value:
                return member
        raise ValueError(f'No member of DimensionalMode has the integer value {value}')


@dataclass(frozen=True)
class Format:
    modes: Tuple[Mode, ...]
    ordering: Tuple[int, ...]

    def __post_init__(self):
        if len(self.modes) != len(self.ordering):
            raise ValueError(f'Length of modes ({len(self.modes)}) must be equal to length of ordering '
                             f'({len(self.ordering)})')

    @property
    def order(self):
        return len(self.modes)

    def deparse(self):
        if self.ordering == tuple(range(self.order)):
            return ''.join(mode.character for mode in self.modes)
        else:
            return ''.join(mode.character + str(ordering) for mode, ordering in zip(self.modes, self.ordering))
