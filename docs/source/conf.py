# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
import swiftsimio


# -- Project information -----------------------------------------------------

project = "SWIFTsimIO"
copyright = "2019, Josh Borrow"
author = "Josh Borrow"

# The full version, including alpha/beta/rc tags
release = swiftsimio.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = [".rst", ".md"]
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Options for automatic API doc

autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
autosummary_generate = True

# must be outside run_apidoc definition to be set successfully:
os.environ["SPHINX_APIDOC_OPTIONS"] = "members,undoc-members,show-inheritance"


def run_apidoc(_):
    try:
        from sphinx.ext.apidoc import main
    except ImportError:
        from sphinx.apidoc import main

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    api_doc_dir = os.path.join(cur_dir, "modules")
    module = os.path.join(cur_dir, "../..", "swiftsimio")
    ignore = [
        os.path.join(cur_dir, "../..", "tests"),
        os.path.join(cur_dir, "../..", "swiftsimio/metadata"),
    ]

    main(["-M", "-f", "-e", "-T", "-d 0", "-o", api_doc_dir, module, *ignore])


def setup(app):
    app.connect("builder-inited", run_apidoc)


intersphinx_mapping = dict(
    numpy=("https://numpy.org/doc/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    unyt=("https://unyt.readthedocs.io/en/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    swiftgalaxy=("https://swiftsimio.readthedocs.io/en/latest/", None),
    velociraptor=("https://velociraptor-python.readthedocs.io/en/latest/", None),
)
