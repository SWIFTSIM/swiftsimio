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

   data = load("cosmo_volume_example.hdf5")

   # This creates a grid that has units msun / Mpc^3, and can be transformed like
   # any other unyt quantity.
   mass_grid = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       periodic=True,
   )

This basic demonstration creates a mass density cube.

To create, for example, a projected temperature cube, we need to remove the
density dependence (i.e. :meth:`render_gas` returns a volumetric
temperature in units of K / kpc^3 and we just want K) by dividing out by
this:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       periodic=True,
   )

   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data,
       resolution=256,
       project="mass_weighted_temps",
       parallel=True,
       periodic=True,
   )

   # A 256 x 256 x 256 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube

Periodic boundaries
-------------------

Cosmological simulations and many other simulations use periodic boundary
conditions. This has implications for the particles at the edge of the
simulation box: they can contribute to voxels on multiple sides of the image.
If this effect is not taken into account, then the voxels close to the edge
will have values that are too low because of missing contributions.

All visualisation functions by default assume a periodic box. Rather than
simply summing each individual particle once, eight additional periodic copies
of each particle are also taken into account. Most copies will contribute
outside the valid voxel range, but the copies that do not ensure that voxels
close to the edge receive all necessary contributions. Thanks to Numba
optimisations, the overhead of these additional copies is relatively small.

There are some caveats with this approach. If you try to visualise a subset of
the particles in the box (e.g. using a mask), then only periodic copies of
particles in this subset will be used. If the subset does not include particles
on the other side of the periodic boundary, then these will still be missing
from the voxel cube. The same is true if you visualise a region of the box.
The periodic boundary wrapping is also not compatible with rotations (see below)
and should therefore not be used together with a rotation.

Rotations
---------

Rotations of the box prior to volume rendering are provided in a similar fashion 
to the :mod:`swiftsimio.visualisation.projection` sub-module, by using the 
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
   from swiftsimio.visualisation.volume_render import render_gas
   from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Specify the rotation parameters
   center = 0.5 * data.metadata.boxsize
   rotate_vec = [0.5,0.5,1]
   matrix = rotation_matrix_from_vector(rotate_vec, axis='z')
   
   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=256,
       project="masses",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True,
       periodic=False, # disable periodic boundaries for rotations
   )
   
   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data, 
       resolution=256,
       project="mass_weighted_temps",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True,
       periodic=False,
   )

   # A 256 x 256 x 256 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube

Lower-level API
---------------

The lower-level API for volume rendering allows for any general positions,
smoothing lengths, and smoothed quantities, to generate a pixel grid that
represents the smoothed, volume rendered, version of the data.

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

Optionally, you will also need:
+ the size of the simulation box in x, y and z, ``box_x``, ``box_y`` and ``box_z``.

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

If the optional arguments ``box_x``, ``box_y`` and ``box_z`` are provided, they
should contain the simulation box size in the same re-scaled coordinates as 
``x``, ``y`` and ``z``. The rendering function will then correctly apply
periodic boundary wrapping. If ``box_x``, ``box_y`` and ``box_z`` are not
provided or set to 0, no periodic boundaries are applied
