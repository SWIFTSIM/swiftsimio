.. SWIFTsimIO documentation master file, created by
   sphinx-quickstart on Sat Nov 23 15:40:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SWIFTsimIO's documentation!
======================================

``swiftsimio`` is a toolkit for reading SWIFT_ data, an astrophysics
simulation code. It is used to ensure that everything that you read has a
symbolic unit attached, and  can be used for visualisation. The final key
feature that it enables is the use of the cell metadata in ``SWIFT``
snapshots to enable partial reading.

.. toctree::
   :maxdepth: 2

   getting_started/index
   loading_data/index
   masking/index
   visualisation/index
   velociraptor/index
   creating_initial_conditions/index
   statistics/index
   command_line/index

   modules/index


Citing SWIFTsimIO
=================

Please cite ``swiftsimio`` using the JOSS paper_:

.. code-block:: bibtex

    @article{Borrow2020,
      doi = {10.21105/joss.02430},
      url = {https://doi.org/10.21105/joss.02430},
      year = {2020},
      publisher = {The Open Journal},
      volume = {5},
      number = {52},
      pages = {2430},
      author = {Josh Borrow and Alexei Borrisov},
      title = {swiftsimio: A Python library for reading SWIFT data},
      journal = {Journal of Open Source Software}
    }


.. _SWIFT: http://www.swiftsim.com
.. _paper: https://joss.theoj.org/papers/10.21105/joss.02430

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
