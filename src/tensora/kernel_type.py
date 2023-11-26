__all__ = ["KernelType"]

from enum import Enum


class KernelType(str, Enum):
    # Python 3.10 does not support StrEnum, so do it manually
    assemble = "assemble"
    compute = "compute"
    evaluate = "evaluate"

    def is_assemble(self):
        return self == KernelType.assemble or self == KernelType.evaluate

    def is_compute(self):
        return self == KernelType.compute or self == KernelType.evaluate

    def __str__(self) -> str:
        return self.name
