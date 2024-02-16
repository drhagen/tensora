__all__ = ["TensorCompiler", "generate_c_code"]

from enum import Enum

from returns.result import Result

from ..kernel_type import KernelType
from ..problem import Problem
from ._taco import generate_c_code_taco
from ._tensora import generate_c_code_tensora


class TensorCompiler(str, Enum):
    # Python 3.10 does not support StrEnum, so do it manually
    tensora = "tensora"
    taco = "taco"

    def __str__(self) -> str:
        return self.name


def generate_c_code(
    problem: Problem, kernel_types: list[KernelType], tensor_compiler: TensorCompiler
) -> Result[str, Exception]:
    match tensor_compiler:
        case TensorCompiler.tensora:
            return generate_c_code_tensora(problem, kernel_types)
        case TensorCompiler.taco:
            return generate_c_code_taco(problem, kernel_types)
