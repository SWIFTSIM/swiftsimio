Contributing to SWIFTsimIO
==========================

Contributions for SWIFTsimIO should come as pull requests submitted through our [GitHub repository](https://github.com/swiftsim/swiftsimio).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions pass all automated tests (you can check this with `pytest`).
+ Your contributions add tests for new functionality.
+ Your contributions are formatted with the `ruff` formatter.
+ Your contributions pass `ruff` style checks.
+ Your contributions are documented with docstrings and their style passes `numpydoc lint` checks.
+ Your contributions are documented fully under `/docs`.

You should also abide by the [code of conduct](https://github.com/SWIFTSIM/swiftsimio/tree/main?tab=coc-ov-file).

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

Ruff
----

You can install the `ruff` linter with `pip install ruff`. To check that your copy of the repository conforms to style rules you can run `ruff check` in the same directory as the `pyproject.toml` file. A message like `All tests passed!` indicates that your working copy passes the checks, otherwise a list of problems is given. Some might be automatically fixable with `ruff check --fix`. Don't forget to commit any automatic fixes.

`ruff` is also used to enforce code formatting, you can check this with `ruff format --check` and automatically format your copy of the code with `ruff format`. Again remember to commit any automatically formatted files.

Pytest unit testing
-------------------

You can install the `pytest` unit testing toolkit with `pip install pytest`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with python versions `3.10`, `3.11`, `3.12` and `3.13`, and the latest PyPI releases of the relevant dependencies (h5py, unyt, etc.). To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Documentation
-------------

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. All public (i.e. not starting in `_`) modules, functions, classes, methods, etc. should have an appropriate docstring. Tests should also have descriptive docstrings, but full descriptions (e.g. of all parameters) are not required.

In addition to this there is "narrative documentation" that should describe the features of the code. The docs are built with `sphinx` and use the "ReadTheDocs" theme. If you have the dependencies installed (check `/docs/requirements.txt`) you can build the documentation locally with `make html` in the `/docs` directory. Opening the `/docs/index.html` file with a browser will then allow you to browse the documentation and check your contributions.

Docstrings
----------

Ruff currently has limited support for [numpydoc](https://numpydoc.readthedocs.io/en/latest/index.html)-style docstrings. To run additional checks on docstrings use `numpydoc lint **/*.py` in the same directory as the `pyproject.toml` file. As more style rules become supported by `ruff` this will hopefully be phased out.
