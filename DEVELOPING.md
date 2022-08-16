# Requirements
* Python 3.8+
* Poetry (`curl -sSL https://install.python-poetry.org | python`)

For the other requirements, inspect the ``pyproject.toml`` file. If you are updated the dependencies, please run `poetry update` to update the

# Development Tasks
There are a series of top-level tasks available through Poetry. These can each be run via

 `poetry run poe <taskname>`

### Basic Verification
* **apply_format** - runs the suite of formatting tools applying tools to make code compliant
* **check_format** - runs the suite of formatting tools checking for compliance
* **lint** - runs the suite of linting tools
* **unit_test** - executes fast unit tests
* **typecheck** - performs static typechecking of the codebase using mypy
* **verify** - executes the basic PR verification suite, which includes all the tasks listed above

### Longer Verification
* **integration_test** - runs slower tests and end-to-end tests are run through this task

### Docsite
* **start_docs** - start the API documentation site locally
* **build_docs** - build the API documentation site

For convenience you can run a check for all style components necessary:

        make run-checks

    This will run isort, black, flake8, mypy, check-manifest, and pydocstyle on the entire repository. Please fix your errors if you see any.

    First, you should run [`isort`](https://github.com/PyCQA/isort) and [`black`](https://github.com/psf/black) to make sure you code is formatted consistently.
    Many IDEs support code formatters as plugins, so you may be able to setup isort and black to run automatically every time you save.
    For example, [`black.vim`](https://github.com/psf/black/tree/master/plugin) will give you this functionality in Vim. But both `isort` and `black` are also easy to run directly from the command line.
    Just run this from the root of your clone:

        isort .
        black .

    Our CI also uses [`flake8`](https://github.com/py-why/dodiscover/tree/main/tests) to lint the code base and [`mypy`](http://mypy-lang.org/) for type-checking. You should run both of these next with

        flake8 .

    and

        mypy .

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/py-why/dodiscover/tree/main/tests). These tests are run with [`pytest`](https://docs.pytest.org/en/latest/), which you can use to locally run any test modules that you've added or changed.

    For example, if you've fixed a bug in `mne_icalabel/a/b.py`, you can run the tests specific to that module with

        pytest -v tests/a/b_test.py

    Our CI will automatically check that test coverage stays above a certain threshold (around 90%). To check the coverage locally in this example, you could run

        pytest -v --cov mne_icalabel.a.b tests/a/b_test.py

    If your contribution involves additions to any public part of the API, we require that you write docstrings
    for each function, method, class, or module that you add.
    See the [Writing docstrings](#writing-docstrings) section below for details on the syntax.
    You should test to make sure the API documentation can build without errors by running

        cd doc
        make html

    If the build fails, it's most likely due to small formatting issues. If the error message isn't clear, feel free to comment on this in your pull request.