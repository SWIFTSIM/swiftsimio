Visualisation
=============

:mod:`swiftsimio` provides visualisation routines in the
:mod:`swiftsimio.visualisation` sub-module.  They are accelerated with the
:mod:`numba` module. They can work without :mod:`numba`, but we strongly recommend
installing it for the best performance (1000x+ speedups).

The four built-in rendering types (described below) have the following
common interface:

.. code-block:: python

   {render_func_name}_gas(
       data=data, # SWIFTsimIO dataset
       resolution=1024, # Resolution along one axis of the output image
       project="masses", # Variable to project, e.g. masses, temperatures, etc.
       parallel=False, # Construct the image in (thread) parallel?
       region=None, # None, or a list telling which region to render_func_name
       periodic=True, # Whether or not to apply periodic boundary conditions
   )

The output of these functions comes with associated units and has the correct
dimensions. There are lower-level APIs (also documented here) that provide
additional functionality.

.. toctree::
   :maxdepth: 2

   projection
   slice
   volume_render
   ray_trace
   power_spectra
   tools

