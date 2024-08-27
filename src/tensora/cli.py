__all__ = ["app"]

from pathlib import Path
from typing import Annotated, Optional

import typer
from parsita import ParseError
from returns.result import Failure, Success

from .expression import parse_assignment
from .format import parse_named_format
from .generate import Language, TensorCompiler, generate_code
from .kernel_type import KernelType
from .problem import make_problem

app = typer.Typer()


@app.command()
def tensora(
    assignment: Annotated[
        str,
        typer.Argument(
            show_default=False,
            help="The assignment for which to generate code, e.g. y(i) = A(i,j) * x(j).",
        ),
    ],
    target_format_strings: Annotated[
        list[str],
        typer.Option(
            "--format",
            "-f",
            help=(
                "A tensor and its format separated by a colon, e.g. A:d1s0 for CSC matrix. "
                "Unmentioned tensors are be assumed to be all dense."
            ),
        ),
    ] = [],  # noqa: B006; Typer does not support Sequence or tuple
    kernel_types: Annotated[
        list[KernelType],
        typer.Option(
            "--type",
            "-t",
            help="The type of kernel that will be generated. Can be mentioned multiple times.",
        ),
    ] = [KernelType.compute],  # noqa: B006; Typer does not support Sequence or tuple
    tensor_compiler: Annotated[
        TensorCompiler,
        typer.Option(
            "--compiler",
            "-c",
            help="The tensor algebra compiler to use to generate the kernel.",
        ),
    ] = TensorCompiler.tensora,
    language: Annotated[
        Language,
        typer.Option(
            "--language",
            "-l",
            help="The language in which to generate the kernel.",
        ),
    ] = Language.c,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            writable=True,
            help=(
                "The file to which the kernel will be written. If not specified, prints to "
                "standard out."
            ),
        ),
    ] = None,
):
    # Parse assignment
    match parse_assignment(assignment):
        case Failure(error):
            typer.echo(f"Failed to parse assignment:\n{error}", err=True)
            raise typer.Exit(1)
        case Success(parsed_assignment):
            pass
        case _:
            raise NotImplementedError()

    # Parse formats
    parsed_formats = {}
    for target_format_string in target_format_strings:
        match parse_named_format(target_format_string):
            case Failure(ParseError(_) as error):
                typer.echo(f"Failed to parse format:\n{error}", err=True)
                raise typer.Exit(1)
            case Failure(error):
                typer.echo(str(error), err=True)
                raise typer.Exit(1)
            case Success((target, format)):
                pass
            case _:
                raise NotImplementedError()

        if target in parsed_formats:
            typer.echo(f"Format for {target} was mentioned multiple times", err=True)
            raise typer.Exit(1)

        parsed_formats[target] = format

    # Validate and standardize assignment and formats
    match make_problem(parsed_assignment, parsed_formats):
        case Failure(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(1)
        case Success(problem):
            pass
        case _:
            raise NotImplementedError()

    # Generate code
    match generate_code(problem, kernel_types, tensor_compiler, language):
        case Failure(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(1)
        case Success(code):
            pass
        case _:
            raise NotImplementedError()

    if output_path is None:
        typer.echo(code)
    else:
        output_path.write_text(code)
