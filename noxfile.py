from nox import options, Session, session

options.sessions = ["test", "test_numpy", "coverage", "lint"]


@session(python=["3.10", "3.11"])
def test(s: Session):
    s.install(".", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests")


@session(python=["3.10", "3.11"])
def test_numpy(s: Session):
    s.install(".[numpy]", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.pandas.{s.python}"
    s.run("python", "-m", "pytest", "--cov", "tensora", "tests/test_numpy.py")


@session(venv_backend="none")
def coverage(s: Session):
    s.run("coverage", "combine")
    s.run("coverage", "html")
    s.run("coverage", "xml")


@session(venv_backend="none")
def lint(s: Session) -> None:
    s.run("ruff", "check", ".")
