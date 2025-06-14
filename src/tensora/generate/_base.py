__all__ = ["TensorCompiler", "generate_code"]

from dataclasses import dataclass
from enum import Enum

from returns.result import Failure, Result, Success

from ..codegen import ir_to_c, ir_to_llvm
from ..kernel_type import KernelType
from ..problem import Problem
from ._taco import generate_c_code_taco
from ._tensora import generate_module_tensora


class TensorCompiler(str, Enum):
    """The tool to generate the tensor kernel.

    Attributes
    ----------
    tensora
        The tensor compiler included with Tensora.
    taco
        The original TACO binary repackaged in `tensora-taco`.
        Only available via the `taco` extra.
        Not compatible with `Language.llvm`.
        Not available on Windows.
    """

    # Python 3.10 does not support StrEnum, so do it manually
    tensora = "tensora"
    taco = "taco"

    def __str__(self) -> str:
        return self.name


class Language(str, Enum):
    """The language to be generated.

    Attributes
    ----------
    c
        C language.
    llvm
        LLVM IR.
        Not compatible with `TensorCompiler.taco`.
    """

    # Python 3.10 does not support StrEnum, so do it manually
    c = "c"
    llvm = "llvm"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class UnsupportedLanguageError(Exception):
    tensor_compiler: TensorCompiler
    language: Language

    def __str__(self):
        return f"For tensor compiler {self.tensor_compiler}, language {self.language} is not supported."


def generate_code(
    problem: Problem,
    kernel_types: list[KernelType],
    tensor_compiler: TensorCompiler,
    language: Language,
) -> Result[str, Exception]:
    match tensor_compiler:
        case TensorCompiler.tensora:
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
        case TensorCompiler.taco:
            match language:
                case Language.c:
                    return generate_c_code_taco(problem, kernel_types)
                case Language.llvm:
                    return Failure(UnsupportedLanguageError(tensor_compiler, language))
