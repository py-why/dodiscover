[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/dodiscover/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/dodiscover/tree/main)
[![unit-tests](https://github.com/py-why/dodiscover/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/dodiscover/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/dodiscover/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/dodiscover)

# What makes dodiscovery different from other causal discovery libraries?

Why do we need another causal discovery library?  Here are a few design elements that differentiate dodiscover from other causal discovery libraries.

## Ease of use

An analyst should be able to get up and running quickly with causal discovery without having to know algorithmic details.
dodiscover prioritizes the workflow over the algorithms, and provide defaults to get your workflow up and running quickly.

## Practical 

## Easy interface for causal assumptions

Causal assumptions are required for all causal inferences.
Modern causal effect inference methods have the user articulate their causal assumptions (e.g., in the form causal DAG) and guide identication and estimation based on those assumptions.

In contrast, causal discovery libraries tend that enable identification of causal inferences attempt to make inferences robust to violations of assumptions, such as the presence of latent confounding variables.

## Default assumption of latent variables


In contrast, causal discovery methods

DoDiscover is a Python library for causal discovery (causal structure learning). If one does not have access to a hypothesized causal graph for their situation, then they may use dodiscover to learn causal structure from their data (e.g. in the form of a graph).

# Documentation

See the [development version documentation](https://py-why.github.io/dodiscover/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/dodiscover/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, dodiscover requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * networkx
    * pandas

For explicit graph functionality for representing various causal graphs, such as ADMG, or CPDAGs, you will also need:

    * pywhy-graphs
    * graphs  # this is a development version for PRable MixedEdgeGraph to networkx

For explicitly representing causal graphs, we recommend using `pywhy-graphs` package, but if you have a graph library that adheres to the graph protocols we require, then you can in principle use those graphs.

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install dodiscover is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U dodiscover

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # for graph functionality
    poetry install --extras graph_func

    # if you would like an editable install of dodiscover for dev purposes
    pip install -e .
