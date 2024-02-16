__all__ = ["generate_c_code_taco", "TacoError"]

from dataclasses import dataclass
from pathlib import Path

from returns.result import Failure, Result, Success

from ..kernel_type import KernelType
from ..problem import Problem
from ._deparse_to_taco import deparse_to_taco


@dataclass(frozen=True, slots=True)
class TacoError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


taco_binary = Path(__file__).parent.joinpath("taco/bin/taco")


def generate_c_code_taco(
    problem: Problem, kernel_types: list[KernelType]
) -> Result[str, TacoError]:
    from tensora_taco import taco_cli

    formats = problem.formats

    expression_string = deparse_to_taco(problem.assignment)
    format_string_arguments = []
    for name, format in formats.items():
        if format.order != 0:  # Taco does not like formats for scalars
            mode_string = "".join(mode.character for mode in format.modes)
            ordering_string = ",".join(map(str, format.ordering))
            format_string_arguments.append(f"-f={name}:{mode_string}:{ordering_string}")

    kernel_type_arguments = []
    for kernel_type in kernel_types:
        match kernel_type:
            case KernelType.evaluate:
                kernel_type_arguments.append("-print-evaluate")
            case KernelType.compute:
                kernel_type_arguments.append("-print-compute")
            case KernelType.assemble:
                kernel_type_arguments.append("-print-assembly")

    taco_arguments = (
        [taco_binary, expression_string, "-print-nocolor"]
        + kernel_type_arguments
        + format_string_arguments
    )

    match taco_cli(taco_arguments):
        case Success(code):
            return Success(code)
        case Failure(message):
            return Failure(TacoError(message))
