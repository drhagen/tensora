import platform

from nox import Session, options, parametrize
from nox_uv import session

options.default_venv_backend = "uv"
options.sessions = ["test", "test_taco", "test_cffi", "test_numpy", "coverage", "lint"]


@session(python=["3.10", "3.11", "3.12", "3.13"], uv_groups=["test"])
def test(s: Session):
    coverage_file = f".coverage.{platform.machine()}.{platform.system()}.{s.python}"
    s.run("coverage", "run", "--data-file", coverage_file, "-m", "pytest", "tests")


@session(python=["3.10", "3.11", "3.12", "3.13"], uv_groups=["test"], uv_extras=["numpy"])
def test_numpy(s: Session):
    coverage_file = f".coverage.{platform.machine()}.{platform.system()}.{s.python}.numpy"
    s.run("coverage", "run", "--data-file", coverage_file, "-m", "pytest", "tests/test_numpy.py")


@session(python=["3.10", "3.11", "3.12", "3.13"], uv_groups=["test"], uv_extras=["cffi"])
def test_cffi(s: Session):
    coverage_file = f".coverage.{platform.machine()}.{platform.system()}.{s.python}.cffi"
    s.run("coverage", "run", "--data-file", coverage_file, "-m", "pytest", "tests_cffi")


@session(python=["3.10", "3.11", "3.12", "3.13"], uv_groups=["test"], uv_extras=["taco"])
def test_taco(s: Session):
    coverage_file = f".coverage.{platform.machine()}.{platform.system()}.{s.python}.taco"
    s.run("coverage", "run", "--data-file", coverage_file, "-m", "pytest", "tests/taco")


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
