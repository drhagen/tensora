__all__ = ["KernelType"]

from enum import StrEnum, auto


class KernelType(StrEnum):
    assembly = auto()
    compute = auto()
    evaluate = auto()

    def is_assembly(self):
        return self == KernelType.assembly or self == KernelType.evaluate
