# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

import sphinx_gallery  # noqa: F401
from sphinx_gallery.sorting import ExampleTitleSortKey

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../"))

import dodiscover  # noqa: E402

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "dodiscover")))

# -- Project information -----------------------------------------------------

project = "dodiscover"
copyright = f"{datetime.today().year}, Adam Li"
author = "Adam Li"
version = dodiscover.__version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_issues",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# generate autosummary even if no references
# -- sphinx.ext.autosummary
autosummary_generate = True

autodoc_default_options = {"inherited-members": None}
autodoc_typehints = "signature"

# -- numpydoc
# Below is needed to prevent errors
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
numpydoc_validate = True

numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    "self.verbose",
    "py",
    "the",
    "functions",
    "lambda",
    "container",
    "iterator",
    "keyword",
    "arguments",
    "dictionary",
    "no",
    "attributes",
    "DAG",
    "causal",
    "CPDAG",
    "PAG",
    "ADMG",
    # networkx
    "node",
    "nodes",
    "graph",
    # numpy
    "ScalarType",
    "ArrayLike",
    # shapes
    "n_times",
    "obj",
    "arrays",
    "lists",
    "func",
    "n_nodes",
    "n_estimated_nodes",
    "n_samples",
    "n_variables",
    "n_classes",
    "NDArray",
    "n_samples_X",
    "n_samples_Y",
    "n_features",
    "n_features_z",
    "k",
    "n_features_X",
    "n_features_Y",
    # deep learning
    "pytorch",
    "model",
}
numpydoc_xref_aliases = {
    # Networkx
    "nx.Graph": "networkx.Graph",
    "nx.DiGraph": "networkx.DiGraph",
    "nx.MultiDiGraph": "networkx.MultiDiGraph",
    "pgmpy.models.BayesianNetwork": "pgmpy.models.BayesianNetwork",
    # dodiscover
    "ADMG": "dodiscover.ADMG",
    "PAG": "dodiscover.PAG",
    "CPDAG": "dodiscover.CPDAG",
    "DAG": "dodiscover.DAG",
    "BaseConditionalIndependenceTest": "dodiscover.ci.BaseConditionalIndependenceTest",
    "Context": "dodiscover.context.Context",
    "PC": "dodiscover.PC",
    "EquivalenceClass": "dodiscover.EquivalenceClass",
    "Graph": "dodiscover.Graph",
    "Column": "dodiscover.typing.Column",
    "NetworkxGraph": "dodiscover.typing.NetworkxGraph",
    "SeparatingSet": "dodiscover.typing.SeparatingSet",
    "ContextBuilder": "dodiscover.context_builder.ContextBuilder",
    # joblib
    "joblib.Parallel": "joblib.Parallel",
    # numpy
    "NDArray": "numpy.ndarray",
    "ArrayLike": ":term:`array_like`",
    # pandas
    "pd.DataFrame": "pandas.DataFrame",
    "column": "pandas.DataFrame.columns",
}

default_role = "py:obj"

# Tell myst-parser to assign header anchors for h1-h3.
# myst_heading_anchors = 3
# suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "auto_examples/index.rst",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "auto_examples/*.rst",
]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "networkx": ("https://networkx.org/documentation/latest/", None),
    "pgmpy": ("https://pgmpy.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}
intersphinx_timeout = 5

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Clean up sidebar: Do not show "Source" link
# html_show_sourcelink = False
# html_copy_source = False

html_theme = "pydata_sphinx_theme"

html_title = f"dodiscover v{version}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/pywhy/dodiscover",
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}

scrapers = ("matplotlib",)

sphinx_gallery_conf = {
    "doc_module": "dodiscover",
    "reference_url": {
        "dodiscover": None,
    },
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "within_subsection_order": ExampleTitleSortKey,
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": "^((?!sgskip).)*$",
    "matplotlib_animations": True,
    "compress_images": ("images", "thumbnails"),
    "image_scrapers": scrapers,
}

# prevent jupyter notebooks from being run even if empty cell
# nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html"],
}

html_context = {
    "versions_dropdown": {
        "dev": "v0.1 (devel)",
    },
}

# Enable nitpicky mode - which ensures that all references in the docs
# resolve.

nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "numpy._typing._array_like._SupportsArray"),
    ("py:class", "numpy._typing._nested_sequence._NestedSequence"),
]


# -- Other extension configuration -------------------------------------------

linkcheck_request_headers = dict(
    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
)  # noqa: E501
linkcheck_anchors = False  # saves a bit of time
linkcheck_timeout = 15  # some can be quite slow
linkcheck_ignore = [
    "https://github.com/{group}/{project}/pull/",  # sphinx-issues will render this properly
]
