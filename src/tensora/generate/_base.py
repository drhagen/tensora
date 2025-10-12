__all__ = ["Language", "generate_code"]

from enum import Enum

from returns.result import Failure, Result, Success

from ..codegen import ir_to_c, ir_to_llvm
from ..kernel_type import KernelType
from ..problem import Problem
from ._tensora import generate_module_tensora


class Language(str, Enum):
    """The language to be generated.

    Attributes
    ----------
    c
        C language.
    llvm
        LLVM IR.
    """

    # Python 3.10 does not support StrEnum, so do it manually
    c = "c"
    llvm = "llvm"

    def __str__(self) -> str:
        return self.name


def generate_code(
    problem: Problem,
    kernel_types: list[KernelType],
    language: Language,
) -> Result[str, Exception]:
    match generate_module_tensora(problem, kernel_types):
        case Failure(error):
            return Failure(error)
        case Success(module):
            match language:
                case Language.c:
                    return Success(ir_to_c(module))
                case Language.llvm:
                    return Success(str(ir_to_llvm(module)))
        case _:
            raise NotImplementedError()
