Volume Rendering
================

The :mod:`swiftsimio.visualisation.volume_render` sub-module provides an
interface to render SWIFT data onto a fixed grid. This takes your 3D data and
finds the 3D density at fixed positions, allowing it to be used in codes that
require fixed grids such as radiative transfer programs.

This effectively solves the equation:

:math:`\tilde{A}_i = \sum_j A_j W_{ij, 3D}`

with :math:`\tilde{A}_i` the smoothed quantity in pixel :math:`i`, and
:math:`j` all particles in the simulation, with :math:`W` the 3D kernel.
Here we use the Wendland-C2 kernel.

The primary function here is
:meth:`swiftsimio.visualisation.volume_render.render_gas`, which allows you
to create a gas density grid of any field, see the example below.

Example
-------

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("my_snapshot_0000.hdf5")

   # This creates a grid that has units msun / Mpc^3, and can be transformed like
   # any other unyt quantity.
   mass_grid = slice_gas(
       data,
       resolution=1024,
       project="masses",
       parallel=True
   )

This basic demonstration creates a mass density cube.

To create, for example, a projected temperature cube, we need to remove the
density dependence (i.e. :meth:`render_gas` returns a volumetric
temperature in units of K / kpc^3 and we just want K) by dividing out by
this:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("my_snapshot_0000.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=1024,
       project="masses",
       parallel=True
   )

   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data,
       resolution=1024,
       project="mass_weighted_temps",
       parallel=True
   )

   # A 1024 x 1024 x 1024 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube

Rotations
---------

Rotations of the box prior to slicing are provided in a similar fashion to the 
:mod:`swiftsimio.visualisation.projection` sub-module, by using the 
:mod:`swiftsimio.visualisation.rotation` sub-module. To rotate the perspective
prior to slicing a ``rotation_center`` argument in :meth:`render_gas` needs
to be provided, specifying the point around which the rotation takes place. 
The angle of rotation is specified with a matrix, supplied by ``rotation_matrix``
in :meth:`render_gas`. The rotation matrix may be computed with 
:meth:`rotation_matrix_from_vector`. This will result in the perspective being 
rotated to be along the provided vector. This approach to rotations applied to 
the above example is shown below.

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.slice import render_gas
   from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

   data = load("my_snapshot_0000.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Specify the rotation parameters
   center = 0.5 * data.metadata.boxsize
   rotate_vec = [0.5,0.5,1]
   matrix = rotation_matrix_from_vector(rotate_vec, axis='z')
   
   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=1024,
       project="masses",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True
   )
   
   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data, 
       resolution=1024,
       project="mass_weighted_temps",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True
   )

   # A 1024 x 1024 x 1024 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube

Lower-level API
---------------

The lower-level API for slices allows for any general positions,
smoothing lengths, and smoothed quantities, to generate a pixel grid that
represents the smoothed, sliced, version of the data.

This API is available through
:meth:`swiftsimio.visualisation.volume_render.scatter` and
:meth:`swiftsimio.visualisation.volume_render.scatter_parallel` for the parallel
version. The parallel version uses significantly more memory as it allocates
a thread-local image array for each thread, summing them in the end. Here we
will only describe the ``scatter`` variant, but they behave in the exact same way.

To use this function, you will need:

+ x-positions of all of your particles, ``x``.
+ y-positions of all of your particles, ``y``.
+ z-positions of all of your particles, ``z``.
+ A quantity which you wish to smooth for all particles, such as their
  mass, ``m``.
+ Smoothing lengths for all particles, ``h``.
+ The resolution you wish to make your cube at, ``res``.

The key here is that only particles in the domain [0, 1] in x, [0, 1] in y,
and [0, 1] in z. will be visible in the cube. You may have particles outside
of this range; they will not crash the code, and may even contribute to the
image if their smoothing lengths overlap with [0, 1]. You will need to
re-scale your data such that it lives within this range. Then you may use the
function as follows:

.. code-block:: python

   from swiftsimio.visualisation.volume_render import scatter

   # Using the variable names from above
   out = scatter(x=x, y=y, z=z, h=h, m=m, res=res)

``out`` will be a 3D :mod:`numpy` grid of shape ``[res, res, res]``. You will
need to re-scale this back to your original dimensions to get it in the
correct units, and do not forget that it now represents the smoothed quantity
per volume.
