from pathlib import Path
from tempfile import TemporaryDirectory

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


def test_multiple_kernels():
    result = runner.invoke(
        app,
        ["y(i) = A(i,j) * x(j)", "-t", "compute", "-t", "evaluate", "-t", "assemble"],
    )

    assert result.exit_code == 0
    assert "compute" in result.stdout
    assert "evaluate" in result.stdout
    assert "assemble" in result.stdout


def test_llvm_language():
    result = runner.invoke(app, ["y(i) = A(i,j) * x(j)", "-l", "llvm"])

    assert result.exit_code == 0
    assert "getelementptr" in result.stdout


def test_write_to_file():
    # Use a temporary directory instead of a temporary file to avoid issue on Windows
    # where the same file cannot be opened by two processes at the same time.
    with TemporaryDirectory() as tmpdir:
        file = Path(tmpdir) / "output.c"
        result = runner.invoke(app, ["y(i) = A(i,j) * x(j)", "-o", str(file)])

        assert result.exit_code == 0
        assert result.stdout == ""
        assert file.read_text().startswith("int32_t compute(taco_tensor_t* restrict y,")


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
def test_bad_input(command):
    result = runner.invoke(app, command, catch_exceptions=False)

    assert result.exit_code == 1
