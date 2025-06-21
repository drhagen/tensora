---
icon: material/hand-heart
---

# Contributing

Tensora is free and open source software developed under an MIT license. Development occurs at the [GitHub project](https://github.com/drhagen/tensora). Contributions, big and small, are welcome.

Bug reports and feature requests may be made directly on the [issues](https://github.com/drhagen/tensora/issues) tab.

To make a pull request, you will need to fork the repo, clone the repo, make the changes, run the tests, push the changes, and [open a PR](https://github.com/drhagen/tensora/pulls).

## Cloning the repo

To make a local copy of Tensora, clone the repository with git:

```shell
git clone https://github.com/drhagen/tensora.git
```

## Installing from source

Tensora uses uv for project and environment management with Hatchling as its build backend. In whatever environment you prefer, ensure [uv](https://github.com/astral-sh/uv) is installed and then use uv to install Tensora and its dependencies:

```shell
uv sync
```

## Testing

Tensora uses pytest to run the tests in the `tests/` directory. The test command is encapsulated with Nox:

```shell
uv run nox -s test test_taco test_cffi test_numpy
```

This will try to test with all compatible Python versions that `nox` can find. To run the tests with only a particular version, run something like this:

```shell
uv run nox -s test-3.10 test_taco-3.10 test_cffi-3.10 test_numpy-3.10
```

It is good to run the tests locally before making a PR, but it is not necessary to have all Python versions run. It is rare for a failure to appear in a single version, and the CI will catch it anyway.

## Code quality

Tensora uses Ruff to ensure a minimum standard of code quality. The code quality commands are encapsulated with Nox:

```shell
uv run nox -s format
uv run nox -s lint
```

## Generating the docs

Tensora uses MkDocs to generate HTML docs from Markdown. For development purposes, they can be served locally without needing to build them first:

```shell
uv run mkdocs serve
```

To deploy the current docs to GitHub Pages, Tensora uses the MkDocs `gh-deploy` command that builds the static site on the `gh-pages` branch, commits, and pushes to the origin:

```shell
uv run mkdocs gh-deploy
```

## Making a release

1. Bump
    1. Increment version in `pyproject.toml`
    2. Run `uv lock`
    3. Commit with message "Bump version number to X.Y.Z"
    4. Push commit to GitHub
    5. Check [CI](https://github.com/drhagen/tensora/actions/workflows/ci.yml) to ensure all tests pass
2. Tag
    1. Tag commit with "vX.Y.Z"
    2. Push tag to GitHub
    3. Wait for [build](https://github.com/drhagen/tensora/actions/workflows/release.yml) to finish
    4. Check [PyPI](https://pypi.org/project/tensora/) for good upload
3. Document
    1. Create [GitHub release](https://github.com/drhagen/tensora/releases) with name "Tensora X.Y.Z" and major changes in body
    2. If appropriate, deploy updated docs
