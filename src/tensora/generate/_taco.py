__all__ = ["generate_c_code_taco", "TacoError"]

import subprocess
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


taco_binary = Path(__file__).parent.parent.joinpath("taco/bin/taco")


def generate_c_code_taco(
    problem: Problem, kernel_types: list[KernelType]
) -> Result[str, TacoError]:
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
        if kernel_type == KernelType.evaluate:
            kernel_type_arguments.append("-print-evaluate")
        elif kernel_type == KernelType.compute:
            kernel_type_arguments.append("-print-compute")
        elif kernel_type == KernelType.assemble:
            kernel_type_arguments.append("-print-assembly")

    # Call taco to write the kernels to standard out
    result = subprocess.run(
        [taco_binary, expression_string, "-print-nocolor"]
        + kernel_type_arguments
        + format_string_arguments,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return Failure(TacoError(result.stderr))

    return Success(result.stdout)
