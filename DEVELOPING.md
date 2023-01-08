# Requirements
* Python 3.8+
* Poetry (`curl -sSL https://install.python-poetry.org | python - --version=1.2.2`)

For the other requirements, inspect the ``pyproject.toml`` file. If you are updated the dependencies, please run `poetry update` to update the

# Development Tasks
There are a series of top-level tasks available through Poetry. These can each be run via

 `poetry run poe <taskname>`

### Basic Verification
* **apply_format** - runs the suite of formatting tools applying tools to make code compliant
* **check_format** - runs the suite of formatting tools checking for compliance
* **lint** - runs the suite of linting tools
* **typecheck** - performs static typechecking of the codebase using mypy
* **unit_test** - executes fast unit tests
* **verify** - executes the basic PR verification suite, which includes all the tasks listed above

### Longer Verification
* **integration_test** - runs slower tests and end-to-end tests are run through this task

### Docsite
* **build_docs** - build the API documentation site

## Details

Here we provide some details to understand the development process.

### Coding Style

For convenience ``poetry`` provides a command line interface for running all the necessary development commands:

    poetry run poe apply_format

This will run isort and black on the entire repository. This will auto-format the code to comply with our coding style.

### Lint

We use linting services to check for common errors in the code.

    poetry run poe lint

We use flake8, bandit, codespell and pydocstyle to check for code smells, which are lines of code that can lead to unintended errors.

### Type checking

We use type checking to check for possible runtime errors due to mismatched types. Python is dynamically typed, so this helps us and the user catch errors that would otherwise then occur during runtime. We use mypy to perform type checking.

    poetry run poe type_check

### Unit tests

In order for any code to be added to the repository, we require unit tests to pass. Any new code should be accompanied by unit tests.

    poetry run poe unit_test

### Integration tests

Dodiscover is part of pywhy's larger ecosystem of causal inference in Python. Because of this tight integration, we also have integration tests, which make sure that any changes in dodiscover do not break or change intended workflows.

    poetry run poe integration_test
