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

and install with ``pip`` using the ``-e`` ("editable") flag,

.. code-block::

   cd swiftsimio
   pip install -e .

Then install the optioanl dependencies for the code and the documentation:

.. code-block::

   pip install -r optional_requirements.txt
   pip install -r docs/requirements.txt

.. include:: ../../../README.rst
   :start-after: COMMUNITY_START_LABEL
   :end-before: COMMUNITY_END_LABEL

Usage example
-------------

.. include:: ../../../README.rst
   :start-after: USAGE_START_LABEL
   :end-before: USAGE_END_LABEL
