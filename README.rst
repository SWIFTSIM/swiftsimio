SWIFTsimIO
==========

|Python version| |PyPI version| |Repostatus| |Build status| |Documentation status| |JOSS| |Black|

.. |Python version| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSWIFTSIM%2Fswiftsimio%2Fmaster%2Fpyproject.toml
   :alt: Supported python versions
.. |PyPI version| image:: https://img.shields.io/pypi/v/swiftsimio
   :target: https://pypi.org/project/swiftsimio
   :alt: Version released on PyPI
.. |Repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Project status: Active - The project has reached a stable, usable state and is being actively developed.
.. |Build status| image:: https://github.com/swiftsim/swiftsimio/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/lint_and_test.yml
   :alt: Build status
.. |Documentation status| image:: https://readthedocs.org/projects/swiftsimio/badge/?version=latest
   :target: https://swiftsimio.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status
.. |JOSS| image:: https://joss.theoj.org/papers/e85c85f49b99389d98f9b6d81f090331/status.svg
   :target: https://joss.theoj.org/papers/e85c85f49b99389d98f9b6d81f090331
   :alt: JOSS publication
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black code style

.. INTRO_START_LABEL

``swiftsimio`` is a toolkit for reading data produced by the SWIFT_
astrophysics simulation code. It is used to ensure that all data have a
symbolic unit attached, and can be used for visualisation. Another key
feature is the use of the cell metadata in ``SWIFT`` snapshots to enable
efficient reading of sub-regions.

The SWIFT_ astrophysical simulation code is used widely. There exists
many ways of reading the data from SWIFT, which outputs HDF5 files.
These range from reading directly using ``h5py`` to using a complex
system such as ``yt``; however these either are unsatisfactory
(e.g. a lack of unit information in reading HDF5), or too complex for
most use-cases. ``swiftsimio`` provides an object-oriented API to
dynamically read data from SWIFT outputs, including FOF and SOAP
catalogues. An extension module for ``swiftsimio`` for using
catalogues and snapshots in tandem is available: ``swiftgalaxy``.

Getting set up with ``swiftsimio`` is easy; it (by design) has very few
requirements. There are a number of optional packages that you can install
to make the experience better and these are recommended.

.. _SWIFT: https://swift.strw.leidenuniv.nl/

.. INTRO_END_LABEL

Full documentation is available at ReadTheDocs_.

.. _ReadTheDocs: http://swiftsimio.readthedocs.org
   
Requirements
------------

.. REQS_START_LABEL

|Python version| is required. Unfortunately it is not
possible to support ``swiftsimio`` on versions of python lower than this.

Python packages
^^^^^^^^^^^^^^^

+ ``numpy``, required for the core numerical routines.
+ ``h5py``, required to read data from the SWIFT HDF5 output files.
+ ``unyt``, required for symbolic unit calculations (depends on sympy``).
+ ``astropy``, required to represent cosmology information.
+ ``numba``, highly recommended should you wish to use the in-built visualisation
  tools.

Optional packages
^^^^^^^^^^^^^^^^^

+ ``scipy``, required if you wish to generate smoothing lengths for particle types
  that do not store this variable in the snapshots (e.g. dark matter)
+ ``tqdm``, required for progress bars for some long-running tasks. If not installed
  no progress bar will be shown.

.. REQS_END_LABEL

Installing
----------

.. INSTALL_START_LABEL
   
``swiftsimio`` can be installed using the ``pip`` python packaging manager,
or any other packaging manager that you wish to use:

.. code-block::

   pip install swiftsimio

.. INSTALL_END_LABEL

Usage example
-------------

.. USAGE_START_LABEL

Example usage is shown below, which plots a density-temperature phase
diagram, with density and temperature given in CGS units:

.. code-block:: python

   import swiftsimio as sw

   # This loads all metadata but explicitly does _not_ read any particle data yet
   data = sw.load("/path/to/swift/output")

   import matplotlib.pyplot as plt

   data.gas.densities.convert_to_cgs()
   data.gas.temperatures.convert_to_cgs()

   plt.loglog()

   plt.scatter(
      data.gas.densities,
      data.gas.temperatures,
      s=1
   )

   plt.xlabel(fr"Gas density $\left[{data.gas.densities.units.latex_repr}\right]$")
   plt.ylabel(fr"Gas temperature $\left[{data.gas.temperatures.units.latex_repr}\right]$")

   plt.tight_layout()

   plt.savefig("test_plot.png", dpi=300)


In the above:

+ All metadata is read in when the ``swiftsimio.load`` function is called.
+ Only the densities and temperatures (corresponding to the ``PartType0/Densities`` and
  ``PartType0/Temperatures``) datasets are read in.
+ That data is only read in once the
  ``swiftsimio.objects.cosmo_array.convert_to_cgs`` method is called.
+ ``swiftsimio.objects.cosmo_array.convert_to_cgs`` converts data in-place;
  i.e. it returns ``None``.
+ The data is cached: it is not re-read when ``plt.scatter`` is called.

.. USAGE_END_LABEL

Citing
------

.. CITING_START_LABEL

Please cite ``swiftsimio`` using the `JOSS paper`_:

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

If you use any of the subsampled projection backends, we ask that you cite our
relevant `SPHERIC paper`_. Citing the arXiv version here is recommended as the
ADS cannot track conference proceedings well.

.. code-block:: bibtex

   @article{Borrow2021,
     title={Projecting SPH Particles in Adaptive Environments}, 
     author={Josh Borrow and Ashley J. Kelly},
     year={2021},
     eprint={2106.05281},
     archivePrefix={arXiv},
     primaryClass={astro-ph.GA}
   }

.. _JOSS paper: https://joss.theoj.org/papers/10.21105/joss.02430
.. _SPHERIC paper: https://arxiv.org/abs/2106.05281

.. CITING_END_LABEL

Community
---------

.. COMMUNITY_START_LABEL

Code contributions are very welcome! A good place to start is the `contributing guide`_ and how to set up a `development environment`_.

``swiftsimio`` is licensed under `GPL-3.0`_ and community members are expected to abide by the `code of conduct`_.

.. _contributing guide: https://github.com/SWIFTSIM/swiftsimio/blob/master/CONTRIBUTING.md
.. _development environment: https://swiftsimio.readthedocs.io/en/latest/getting_started/index.html#installing
.. _GPL-3.0: https://github.com/SWIFTSIM/swiftgalaxy/tree/main?tab=GPL-3.0-1-ov-file
.. _code of conduct: https://github.com/SWIFTSIM/swiftsimio/tree/main?tab=coc-ov-file

.. COMMUNITY_END_LABEL
