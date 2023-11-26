__all__ = ["app"]

from pathlib import Path
from typing import Annotated, Optional

import typer
from parsita import Failure, Success

from .expression import parse_assignment
from .format import Format, Mode, parse_format
from .kernel_type import KernelType
from .native import generate_c_code_from_parsed

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
    ] = [],
    kernel_types: Annotated[
        list[KernelType],
        typer.Option(
            "--type",
            "-t",
            help="The type of kernel that will be generated. Can be mentioned multiple times.",
        ),
    ] = [KernelType.compute],
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
        case Success(sugar):
            sugar = sugar

    # Parse formats
    parsed_formats = {}
    for target_format_string in target_format_strings:
        split_format = target_format_string.split(":")
        if len(split_format) != 2:
            typer.echo(
                f"Format must be of the form 'target:format_string': {target_format_string}",
                err=True,
            )
            raise typer.Exit(1)

        target, format_string = split_format

        if target in parsed_formats:
            typer.echo(f"Format for {target} was mentioned multiple times", err=True)
            raise typer.Exit(1)

        match parse_format(format_string):
            case Failure(error):
                typer.echo(f"Failed to parse format:\n{error}", err=True)
                typer.Exit(1)
            case Success(format):
                parsed_formats[target] = format

    # Fill in missing formats with dense formats
    # Use the order of variable_orders to determine the parameter order
    formats = {}
    for variable_name, order in sugar.variable_orders().items():
        if variable_name in parsed_formats:
            formats[variable_name] = parsed_formats[variable_name]
        else:
            formats[variable_name] = Format((Mode.dense,) * order, tuple(range(order)))

    # Generate code
    code = generate_c_code_from_parsed(sugar, formats, kernel_types)

    if output_path is None:
        typer.echo(code)
    else:
        output_path.write_text(code)
