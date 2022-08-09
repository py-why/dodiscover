[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/dodiscover/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/dodiscover/tree/main)
[![unit-tests](https://github.com/py-why/dodiscover/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/dodiscover/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/dodiscover/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/dodiscover)

# dodiscover

dodiscover is a Python library for causal discovery, or structure learning. This is generally considered a "first step" in the causal inference pipeline, if one does not have access to a hypothesized causal graph for their situation.

# Documentation

See the [development version documentation](https://pywhy.github.io/dodiscover/dev/index.html).

Or see [stable version documentation](https://pywhy.github.io/dodiscover/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, dodiscover requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * networkx
    * pywhy-graphs

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install dodiscover is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U dodiscover

    # If you are a developer and would like to install the developer dependencies
    pip install dodiscover[doc,style,test]

    # If you would like full functionality, which installs all of the above
    pip install dodiscover[all]

To install the package from github, clone the repository and then `cd` into the directory:

    pip install -e .

    # One can also add the different identifiers, such as '[doc]' to install
    # extra dependencies

# Current Limitations and Current Roadmap

Currently, selection bias representation is not implemented in the corresponding algorithms. However, I believe it is technically feasible based on the design of how we use networkx.
