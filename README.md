SWIFTsimIO
==========

[![Build Status](https://travis-ci.com/SWIFTSIM/swiftsimio.svg?branch=master)](https://travis-ci.com/SWIFTSIM/swiftsimio)
[![Documentation Status](https://readthedocs.org/projects/swiftsimio/badge/?version=latest)](http://swiftsimio.readthedocs.org/latest/?badge=latest)

The SWIFT astrophysical simulation code (http://swift.dur.ac.uk) is used
widely. There exists many ways of reading the data from SWIFT, which outputs
HDF5 files. These range from reading directly using `h5py` to using a complex
system such as `yt`; however these either are unsatisfactory (e.g. a lack of
unit information in reading HDF5), or too complex for most use-cases. This
(thin) wrapper provides an object-oriented API to read (dynamically) data
from SWIFT.

Full documentation is available at [ReadTheDocs](http://swiftsimio.readthedocs.org).

Getting set up with `swiftsimio` is easy; it (by design) has very few
requirements. There are a number of optional packages that you can install
to make the experience better and these are recommended.


Requirements
------------

This requires `python` `v3.6.0` or higher. Unfortunately it is not
possible to support `swiftsimio` on versions of python lower than this.
It is important that you upgrade if you are still a `python2` user.

### Python packages


+ `numpy`, required for the core numerical routines.
+ `h5py`, required to read data from the SWIFT HDF5 output files.
+ `unyt`, required for symbolic unit calculations (depends on sympy`).

### Optional packages


+ `numba`, highly recommended should you wish to use the in-built visualisation
  tools.
+ `scipy`, required if you wish to generate smoothing lengths for particle types
  that do not store this variable in the snapshots (e.g. dark matter)
+ `tqdm`, required for progress bars for some long-running tasks. If not installed
  no progress bar will be shown.
+ `py-sphviewer`, if you wish to use our integration with this visualisation
  code.


Installing
----------

`swiftsimio` can be installed using the python packaging manager, `pip`,
or any other packaging manager that you wish to use:

`pip install swiftsimio`