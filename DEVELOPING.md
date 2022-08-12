# Requirements
* Python 3.8+
* Poetry (`curl -sSL https://install.python-poetry.org | python`)

For the other requirements, inspect the ``pyproject.toml`` file. If you are updated the dependencies, please run `poetry update` to update the

# Development Tasks
There are a series of top-level tasks available through Poetry. These can each be run via

 `poetry run poe <taskname>`

### Basic Verification
* **format** - runs the suite of formatting tools
* **lint** - runs the suite of linting tools
* **unit_test** - executes fast unit tests
* **typecheck** - performs static typechecking of the codebase using mypy
* **verify** - executes the basic PR verification suite, which includes all the tasks listed above

### Longer Verification
* **integration_test** - runs slower tests and end-to-end tests are run through this task

### Docsite
* **start_docs** - start the API documentation site locally
* **build_docs** - build the API documentation site