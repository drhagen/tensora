from nox import options, parametrize
from nox_poetry import Session, session

options.sessions = ["test", "test_taco", "test_cffi", "test_numpy", "coverage", "lint"]


@session(python=["3.10", "3.11", "3.12", "3.13"])
def test(s: Session):
    s.install(".", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests")


@session(python=["3.10", "3.11", "3.12", "3.13"])
def test_taco(s: Session):
    s.install(".[taco]", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.taco.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests/taco")


@session(python=["3.10", "3.11", "3.12", "3.13"])
def test_cffi(s: Session):
    s.install(".[cffi]", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.cffi.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests_cffi")


@session(python=["3.10", "3.11", "3.12", "3.13"])
def test_numpy(s: Session):
    s.install(".[numpy]", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.numpy.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests/test_numpy.py")


@session(venv_backend="none")
def coverage(s: Session):
    s.run("coverage", "combine")
    s.run("coverage", "html")
    s.run("coverage", "xml")


@session(venv_backend="none")
def fuzz(s: Session):
    s.run("hypothesis", "fuzz", "fuzz_tests")


@session(venv_backend="none")
@parametrize("command", [["ruff", "check", "."], ["ruff", "format", "--check", "."]])
def lint(s: Session, command: list[str]):
    s.run(*command)


@session(venv_backend="none")
def format(s: Session) -> None:
    s.run("ruff", "check", ".", "--select", "I", "--fix")
    s.run("ruff", "format", ".")
