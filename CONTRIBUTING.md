Contributing to SWIFTsimIO
==========================

Contributions for SWIFTsimIO should come as pull requests submitted through our [GitHub repository](https://github.com/swiftsim/swiftsimio).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions pass all automated tests (you can check this with `pytest`)
+ Your contributions add tests for new functionality
+ Your contributions are formatted with the `black` formatter (see `format.sh`)
+ Your contributions are documented fully under `/docs`.

You should also abide by the [code of conduct](https://github.com/SWIFTSIM/swiftsimio/tree/main?tab=coc-ov-file).

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

Black style
-----------

To check your copy of the repository you can then run `black --check` in the same directory as the `pyproject.toml` file. A message like `All done! ✨ 🍰 ✨` indicates that your working copy passes the checks, while `Oh no! 💥 💔 💥` indicates problems are present. You can also use `black` to automatically edit your copy of the repository to comply with the style rules by running `./format.sh` in the same directory as `pyproject.toml`. Don't forget to commit any changes it makes.

Pytest unit testing
-------------------

You can install the `pytest` unit testing toolkit with `pip install pytest`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with python versions `3.10`, `3.11`, `3.12` and `3.13`, and the latest PyPI releases of the relevant dependencies (h5py, unyt, etc.). To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Documentation
-------------

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. All public (i.e. not starting in `_`) modules, functions, classes, methods, etc. should have an appropriate docstring. In addition to this there is "narrative documentation" that should describe the features of the code. The docs are built with `sphinx` and use the "ReadTheDocs" theme. If you have the dependencies installed (check `/docs/requirements.txt`) you can build the documentation locally with `make html` in the `/docs` directory. Opening the `/docs/index.html` file with a browser will then allow you to browse the documentation and check your contributions.
