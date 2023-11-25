__all__ = ["KernelType"]

from enum import Enum


class KernelType(str, Enum):
    # Python 3.10 does not support StrEnum, so do it manually
    assembly = "assembly"
    compute = "compute"
    evaluate = "evaluate"

    def is_assembly(self):
        return self == KernelType.assembly or self == KernelType.evaluate
