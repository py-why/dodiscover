[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/dodiscover/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/dodiscover/tree/main)
[![unit-tests](https://github.com/py-why/dodiscover/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/dodiscover/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/dodiscover/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/dodiscover)

# DoDiscover

DoDiscover is a Python library for causal discovery (causal structure learning). If one does not have access to a causal graph for their modeling problem, they may use DoDiscover to learn causal structure from their data (e.g., in the form of a graph).

# What makes dodiscover different from other causal discovery libraries?

Why do we need another causal discovery library?
Here are some design goals that differentiate DoDiscover from other causal discovery libraries.

## Ease of use

An analyst should be able to get a causal discovery workflow working quickly without intimate knowledge of causal discovery algorithms.
DoDiscover prioritizes the workflow over the algorithms and provides default arguments to algorithm parameters.

## Democratizing deep causal discovery

Many cutting-edge causal discovery algorithms rely on deep learning frameworks.
However, deep learning-based causal discovery often requires obscure boilerplate code, complex configuration, and management of large artifacts such as embeddings.
DoDiscover seeks to create abstractions that address these challenges and make deep causal discovery more broadly accessible. Current algorithms are a work-in-progress. We will begin by providing a robust API for the fundamental discovery algorithms.

## Easy interface for articulating causal assumptions

Domain experts bring a large amount of domain knowledge to a problem.
That domain knowledge can establish causal assumptions that can constrain causal discovery.
Causal discovery (indeed, all causal inferences) requires causal assumptions.

However, a newly developed causal discovery algorithm has a greater research impact when it can do more with fewer assumptions.
This "do more with less" orientation tends to deemphasize assumptions in the user interfaces of many causal discovery libraries.

DoDiscover prioritizes the interface for causal assumptions.
Further, DoDiscover seeks to help the user feel confident with their assumptions by emphasizing testing assumptions, making inferences under uncertainty, and robustness to model misspecification.

## Unite causal discovery and causal representation learning

DoDiscover is a Python library for causal discovery (causal structure learning).
Our goal is to provide developers and researchers with guide rails for causal discovery that doesn't require deep knowledge of individual causal discovery algorithms.

## What is the difference between dodiscover and other pywhy packages?

The goal of dodiscover is to flatten the on-ramp to causal discovery algorithms.
DoWhy provides a consistent API for various causal tasks that typically require a graph structure.
Similarly, DoDiscover aims to provide a cohesive and user-friendly API to apply causal discovery algorithms for inferring a causal graph from data.

[causal-learn](https://github.com/py-why/causal-learn) is an extensive collection of causal discovery algorithms.
It continuous to host new cutting-edge algorithms in causal discovery.
However, these algorithms do not have a unified API.
Further, the historic focus of causal-learn is increasing the capabilities of discovery algorithms.
In contrast, dodiscover's focus is on the discovery API and usability.

When possible, dodiscover prefers to provide an API wrapper to discovery algorithms in causal-learn and other libraries.
Please consider contributing to [causal-learn](https://github.com/py-why/causal-learn) if you plan to implement an algorithm from scratch, then contributing a wrapper in dodiscover.

In the future we plan on trying to integrate the two libraries.

## What is the relationship with pywhy-graphs and pywhy-stats?

[pywhy-graphs](https://github.com/py-why/pywhy-graphs) is the home of graph data structures and graph algorithms in PyWhy.

[pywhy-stats](https://github.com/py-why/pywhy-stats) serves as a repository for implementations of (un)conditional independence tests, which can be utilized in various tasks, such as causal discovery.

# Documentation

See the [development version documentation](https://py-why.github.io/dodiscover/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/dodiscover/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, dodiscover requires:

    * Python (>=3.10)
    * numpy
    * scipy
    * networkx
    * pandas

We have removed support for Python 3.8 as we depend explicitly on networkx, which has deprecated Python 3.8 support. For explicit graph functionality for representing various causal graphs, such as ADMG, or CPDAGs, you will also need:

    * pywhy-graphs

For explicitly representing causal graphs, we recommend using `pywhy-graphs` package, but if you have a graph library that adheres to the graph protocols we require, then you can in principle use those graphs.

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install dodiscover is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U dodiscover

To install the package from github, clone the repository and then `cd` into the directory. You can then use `pip` to install:

    pip install -e .

    # for extra functionality for documentation, building, style checking and unit-testing
    pip install .[doc, build, style, test]
