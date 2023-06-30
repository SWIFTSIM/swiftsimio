Getting Started
===============

The SWIFT astrophysical simulation code (http://swift.dur.ac.uk) is used
widely. There exists many ways of reading the data from SWIFT, which outputs
HDF5 files. These range from reading directly using :mod:`h5py` to using a
complex system such as :mod:`yt`; however these either are unsatisfactory
(e.g. a lack of unit information in reading HDF5), or too complex for most
use-cases. This (thin) wrapper provides an object-oriented API to read
(dynamically) data from SWIFT.

Getting set up with :mod:`swiftsimio` is easy; it (by design) has very few
requirements. There are a number of optional packages that you can install
to make the experience better and these are recommended. All requirements
are detailed below.


Requirements
------------

This requires ``python`` ``v3.8.0`` or higher. Unfortunately it is not
possible to support :mod:`swiftsimio` on versions of python lower than this.
It is important that you upgrade if you are still a ``python2`` user.

Python packages
^^^^^^^^^^^^^^^

+ ``numpy``, required for the core numerical routines.
+ ``h5py``, required to read data from the SWIFT HDF5 output files.
+ ``unyt``, required for symbolic unit calculations (depends on ``sympy``).

Optional packages
^^^^^^^^^^^^^^^^^

+ ``numba``, highly recommended should you wish to use the in-built visualisation
  tools.
+ ``scipy``, required if you wish to generate smoothing lengths for particle types
  that do not store this variable in the snapshots (e.g. dark matter)
+ ``tqdm``, required for progress bars for some long-running tasks. If not installed
  no progress bar will be shown.


Installing
----------

:mod:`swiftsimio` can be installed using the python packaging manager, ``pip``,
or any other packaging manager that you wish to use:

``pip install swiftsimio``

Note that this will install any required packages for you.

To set up the code for development, first clone the latest master from GitHub:

``git clone https://github.com/SWIFTSIM/swiftsimio.git``

and install with ``pip`` using the ``-e`` flag,

``cd swiftsimio``

``pip install -e .``

.. TODO: Add contribution guide.

Usage
-----

There are many examples of using :mod:`swiftsimio` available in the
swiftsimio_examples_ repository, which also includes examples for reading
older (e.g. EAGLE) datasets.

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


Don't worry too much about this for now if you can't understand it, we will
get into this much more heavily in the next section.

In the above it's important to note the following:

+ All metadata is read in when the :meth:`swiftsimio.load` function is called.
+ Only the density and temperatures (corresponding to the ``PartType0/Densities`` and
  ``PartType0/Temperatures``) datasets are read in.
+ That data is only read in once the
  :meth:`swiftsimio.objects.cosmo_array.convert_to_cgs` method is called.
+ :meth:`swiftsimio.objects.cosmo_array.convert_to_cgs` converts data in-place;
  i.e. it returns `None`.
+ The data is cached and not re-read in when ``plt.scatter`` is called.


.. _swiftsimio_examples: https://github.com/swiftsim/swiftsimio-examples
