__all__ = ["KernelType"]

from enum import Enum, auto


class KernelType(Enum):
    assembly = auto()
    compute = auto()
    evaluate = auto()

    def is_assembly(self):
        return self == KernelType.assembly or self == KernelType.evaluate
