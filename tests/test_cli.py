from pathlib import Path
from tempfile import NamedTemporaryFile

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
        app, ["y(i) = A(i,j) * x(j)", "-t", "compute", "-t", "evaluate", "-t", "assemble"]
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
