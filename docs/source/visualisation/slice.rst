Slices
======

The :mod:`swiftsimio.visualisation.slice` sub-module provides an interface
to render SWIFT data onto a slice. This takes your 3D data and finds the 3D
density at fixed z-position, slicing through the box.

This effectively solves the equation:

:math:`\tilde{A}_i = \sum_j A_j W_{ij, 3D}`

with :math:`\tilde{A}_i` the smoothed quantity in pixel :math:`i`, and
:math:`j` all particles in the simulation, with :math:`W` the 3D kernel.
Here we use the Wendland-C2 kernel. Note that here we take the kernel
at a fixed z-position.

The primary function here is
:meth:`swiftsimio.visualisation.slice.slice_gas`, which allows you to
create a gas slice of any field. See the example below.

Example
-------

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.slice import slice_gas

   data = load("cosmo_volume_example.hdf5")

   # This creates a grid that has units msun / Mpc^3, and can be transformed like
   # any other unyt quantity. Note that `slice` is given in terms of the box-size,
   # so here we are taking a slice at z = boxsize / 2.
   mass_map = slice_gas(
       data,
       slice=0.5,
       resolution=1024,
       project="masses",
       parallel=True
   )

   # Let's say we wish to save it as g / cm^2,
   from unyt import g, cm
   mass_map.convert_to_units(g / cm**3)

   from matplotlib.pyplot import imsave
   from matplotlib.colors import LogNorm

   # Normalize and save
   imsave("gas_slice_map.png", LogNorm()(mass_map.value), cmap="viridis")


This basic demonstration creates a mass density map.

To create, for example, a projected temperature map, we need to remove the
density dependence (i.e. :meth:`slice_gas` returns a volumetric temperature
in units of K / kpc^3 and we just want K) by dividing out by this:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.slice import slice_gas

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Map in msun / mpc^3
   mass_map = slice_gas(
       data,
       slice=0.5,
       resolution=1024,
       project="masses",
       parallel=True
   )

   # Map in msun * K / mpc^3
   mass_weighted_temp_map = slice_gas(
       data,
       slice=0.5,
       resolution=1024,
       project="mass_weighted_temps",
       parallel=True
   )

   temp_map = mass_weighted_temp_map / mass_map

   from unyt import K
   temp_map.convert_to_units(K)

   from matplotlib.pyplot import imsave
   from matplotlib.colors import LogNorm

   # Normalize and save
   imsave("temp_map.png", LogNorm()(temp_map.value), cmap="twilight")


The output from this example, when used with the example data provided in the
loading data section should look something like:

.. image:: temp_slice.png

Rotations
---------

Rotations of the box prior to slicing are provided in a similar fashion to the 
:mod:`swiftsimio.visualisation.projection` sub-module, by using the 
:mod:`swiftsimio.visualisation.rotation` sub-module. To rotate the perspective
prior to slicing a ``rotation_center`` argument in :meth:`slice_gas` needs
to be provided, specifying the point around which the rotation takes place. 
The angle of rotation is specified with a matrix, supplied by ``rotation_matrix``
in :meth:`slice_gas`. The rotation matrix may be computed with 
:meth:`rotation_matrix_from_vector`. This will result in the perspective being 
rotated to be along the provided vector. This approach to rotations applied to 
the above example is shown below.

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.slice import slice_gas
   from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Specify the rotation parameters
   center = 0.5 * data.metadata.boxsize
   rotate_vec = [0.5,0.5,1]
   matrix = rotation_matrix_from_vector(rotate_vec, axis='z')
   
   # Map in msun / mpc^3
   mass_map = slice_gas(
       data,
       slice=0.5,
       resolution=1024,
       project="masses",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True
   )
   
   # Map in msun * K / mpc^3
   mass_weighted_temp_map = slice_gas(
       data, 
       slice=0.5,
       resolution=1024,
       project="mass_weighted_temps",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True
   )

   temp_map = mass_weighted_temp_map / mass_map

   from unyt import K
   temp_map.convert_to_units(K)

   from matplotlib.pyplot import imsave
   from matplotlib.colors import LogNorm

   # Normalize and save
   imsave("temp_map.png", LogNorm()(temp_map.value), cmap="twilight")


Lower-level API
---------------

The lower-level API for slices allows for any general positions,
smoothing lengths, and smoothed quantities, to generate a pixel grid that
represents the smoothed, sliced, version of the data.

This API is available through
:meth:`swiftsimio.visualisation.slice.slice_scatter` and
:meth:`swiftsimio.visualisation.slice.slice_scatter_parallel` for the parallel
version. The parallel version uses significantly more memory as it allocates
a thread-local image array for each thread, summing them in the end. Here we
will only describe the ``scatter`` variant, but they behave in the exact same way.

To use this function, you will need:

+ x-positions of all of your particles, ``x``.
+ y-positions of all of your particles, ``y``.
+ z-positions of all of your particles, ``z``.
+ Where in the [0,1] range you wish to slice, ``z_slice``.
+ A quantity which you wish to smooth for all particles, such as their
  mass, ``m``.
+ Smoothing lengths for all particles, ``h``.
+ The resolution you wish to make your square image at, ``res``.

The key here is that only particles in the domain [0, 1] in x, [0, 1] in y,
and [0, 1] in z. will be visible in the image. You may have particles outside
of this range; they will not crash the code, and may even contribute to the
image if their smoothing lengths overlap with [0, 1]. You will need to
re-scale your data such that it lives within this range. Then you may use the
function as follows:

.. code-block:: python

   from swiftsimio.visualisation.slice import slice_scatter

   # Using the variable names from above
   out = slice_scatter(x=x, y=y, z=z, h=h, m=m, z_slice=z_slice, res=res)

``out`` will be a 2D :mod:`numpy` grid of shape ``[res, res]``. You will need
to re-scale this back to your original dimensions to get it in the correct units,
and do not forget that it now represents the smoothed quantity per volume.
