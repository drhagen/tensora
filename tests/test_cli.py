from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from typer.testing import CliRunner

from tensora.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Options" in result.stdout


def test_cli():
    result = runner.invoke(app, ["y(i) = A(i,j) * x(j)", "-f", "A:ds"])

    assert result.exit_code == 0
    assert result.stdout.startswith("int32_t compute(taco_tensor_t* restrict y,")


@pytest.mark.parametrize("compiler", [[], ["-c", "tensora"], ["-c", "taco"]])
def test_multiple_kernels(compiler):
    result = runner.invoke(
        app,
        ["y(i) = A(i,j) * x(j)", "-t", "compute", "-t", "evaluate", "-t", "assemble"] + compiler,
    )

    assert result.exit_code == 0
    assert "compute" in result.stdout
    assert "evaluate" in result.stdout
    assert "assemble" in result.stdout


def test_write_to_file():
    with NamedTemporaryFile(suffix=".c") as f:
        result = runner.invoke(app, ["y(i) = A(i,j) * x(j)", "-o", f.name])

        assert result.exit_code == 0
        assert result.stdout == ""

        assert Path(f.name).read_text().startswith("int32_t compute(taco_tensor_t* restrict y,")


@pytest.mark.parametrize("compiler", ["tensora", "taco"])
@pytest.mark.parametrize(
    "command",
    [
        ["a(i) = b(i) +"],
        ["y(i) = A(i,j) * x(j)", "-f=ds"],
        ["y(i) = A(i,j) * x(j)", "-f=A:d1s2"],
        ["y(i) = A(i,j) * x(j)", "-f=A:ds", "-f=A:dd"],
        ["y(i) = A(i,j) * x(j)", "-f=A:d"],
        ["y(i) = A(i,j) * x(j)", "-f=B:ds"],
        ["a(i) = A(i,i)"],
        ["A(i,j) = B(i,j) + C(j,i)", "-f=A:ds", "-f=B:ds", "-f=C:ds"],
    ],
)
def test_bad_input(compiler, command):
    if command[0] == "a(i) = A(i,i)" and compiler == "taco":
        pytest.xfail("Taco CLI succeeds but generates invalid code")

    result = runner.invoke(app, command + ["-c", compiler], catch_exceptions=False)

    assert result.exit_code == 1
