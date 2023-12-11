from hypothesis import assume, given
from hypothesis import strategies as st
from typer.testing import CliRunner

from tensora.cli import app

runner = CliRunner()


@given(command=st.lists(st.text()))
def test_cli_cannot_crash(command):
    # Arguments to a CLI cannot contain null bytes.
    assume(not any("\0" in string for string in command))

    _ = runner.invoke(app, command, catch_exceptions=False)
