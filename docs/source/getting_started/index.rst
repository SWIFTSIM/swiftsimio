Getting Started
===============

.. include:: ../../../README.rst
   :start-after: INTRO_START_LABEL
   :end-before: INTRO_END_LABEL

Requirements
------------

.. include:: ../../../README.rst
   :start-after: REQS_START_LABEL
   :end-before: REQS_END_LABEL

.. |Python version| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSWIFTSIM%2Fswiftsimio%2Fmaster%2Fpyproject.toml

Installing
----------

.. include:: ../../../README.rst
   :start-after: INSTALL_START_LABEL
   :end-before: INSTALL_END_LABEL

Development environment
^^^^^^^^^^^^^^^^^^^^^^^

To set up the code for development, first clone the latest master from GitHub:

.. code-block::
   
   git clone https://github.com/SWIFTSIM/swiftsimio.git

and install with ``pip`` using the ``-e`` ("editable") flag, and specifying optional dependencies for development and building the documentation:

.. code-block::

   cd swiftsimio
   pip install -e .[dev,docs]

.. include:: ../../../README.rst
   :start-after: COMMUNITY_START_LABEL
   :end-before: COMMUNITY_END_LABEL

Usage example
-------------

.. include:: ../../../README.rst
   :start-after: USAGE_START_LABEL
   :end-before: USAGE_END_LABEL
