Projection
==========

The :mod:`swiftsimio.visualisation.projection` sub-module provides an interface
to render SWIFT data projected to a grid. This takes your 3D data and projects
it down to 2D, such that if you request masses to be smoothed then these
functions return a surface density.

This effectively solves the equation:

:math:`\tilde{A}_i = \sum_j A_j W_{ij, 2D}`

with :math:`\tilde{A}_i` the smoothed quantity in pixel :math:`i`, and
:math:`j` all particles in the simulation, with :math:`W` the 2D kernel.
Here we use the Wendland-C2 kernel.

The primary function here is
:meth:`swiftsimio.visualisation.projection.project_gas`, which allows you to
create a gas projection of any field. See the example below.

Example
-------

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.projection import project_gas

   data = load("my_snapshot_0000.hdf5")

   # This creates a grid that has units msun / Mpc^2, and can be transformed like
   # any other unyt quantity
   mass_map = project_gas(data, resolution=1024, project="masses", parallel=True)

   # Let's say we wish to save it as msun / kpc^2,
   from unyt import msun, kpc
   mass_map.convert_to_units(msun / kpc**2)

   from matplotlib.pyplot import imsave
   from matplotlib.colors import LogNorm

   # Normalize and save
   imsave("gas_surface_dens_map.png", LogNorm()(mass_map.value), cmap="viridis")


This basic demonstration creates a mass surface density map.

To create, for example, a projected temperature map, we need to remove the
surface density dependence (i.e. :meth:`project_gas` returns a surface
temperature in units of K / kpc^2 and we just want K) by dividing out by
this:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.projection import project_gas

   data = load("my_snapshot_0000.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Map in msun / mpc^2
   mass_map = project_gas(data, resolution=1024, project="masses", parallel=True)
   # Map in msun * K / mpc^2
   mass_weighted_temp_map = project_gas(
       data,
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


Rotations
---------

Sometimes you will need to visualise a galaxy from a different perspective.
The :mod:`swiftsimio.visualisation.rotation` sub-module provides routines to
generate rotation matrices corresponding to vectors, which can then be
provided to the ``rotation_matrix`` argument of :meth:`project_gas` (and
:meth:`project_gas_pixel_grid`). You will also need to supply the
``rotation_center`` argument, as the rotation takes place around this given
point. The example code below loads a snapshot, and a halo catalogue, and
creates an edge-on and face-on projection using the integration in
``velociraptor``.

.. code-block:: python

   from swiftsimio import load, mask
   from velociraptor import load as load_catalogue
   from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
   from swiftsimio.visualisation.projection import project_gas_pixel_grid

   import unyt
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm

   # Radius around which to load data, we will visualise half of this
   size = 100 * unyt.kpc

   snapshot_filename = "eagle.hdf5"
   catalogue_filename = "stf.properties"

   catalogue = load_catalogue(catalogue_filename)

   # Which halo should we visualise?
   halo = 25

   x = catalogue.positions.xcmbp[halo]
   y = catalogue.positions.ycmbp[halo]
   z = catalogue.positions.zcmbp[halo]

   lx = catalogue.angular_momentum.lx_star[halo]
   ly = catalogue.angular_momentum.ly_star[halo]
   lz = catalogue.angular_momentum.lz_star[halo]

   # The angular momentum vector will point perpendicular to the galaxy disk.
   angular_momentum_vector = np.array([lx.value, ly.value, lz.value])
   angular_momentum_vector /= np.linalg.norm(angular_momentum_vector)

   face_on_rotation_matrix = rotation_matrix_from_vector(vector)
   edge_on_rotation_matrix = rotation_matrix_from_vector(vector, axis="y")

   region = [
      [x - size, x + size],
      [y - size, y + size],
      [z - size, z + size],
   ]

   visualise_region = [
      x - 0.5 * size, x + 0.5 * size,
      y - 0.5 * size, y + 0.5 * size,
   ]

   data_mask = mask(snapshot_filename)
   data_mask.constrain_spatial(region)
   data = load(snapshot_filename, mask=data_mask)

   # Use project_gas_pixel_grid to generate projected images

   common_arguments = dict(data=data, resolution=512, parallel=True, region=visualise_region)

   un_rotated = project_gas_pixel_grid(**common_arguments)

   face_on = project_gas_pixel_grid(
      **common_arguments,
      rotation_center=unyt.unyt_array([x, y, z]),
      rotation_matrix=face_on_rotation_matrix,
   )

   edge_on = project_gas_pixel_grid(
      **common_argumetns,
      rotation_center=unyt.unyt_array([x, y, z]),
      rotation_matrix=edge_on_rotation_matrix,
   )


Other particle types
--------------------

Other particle types are able to be visualised through the use of the
:meth:`swiftsimio.visualisation.projection.project_pixel_grid` function. This
does not attach correct symbolic units, so you will have to work those out
yourself, but it does perform the smoothing. We aim to introduce the feature
of correctly applied units to these projections soon.

To use this feature for particle types that do not have smoothing lengths, you
will need to generate them, as in the example below where we create a
mass density map for dark matter. We provide a utility to do this through
:meth:`swiftsimio.visualisation.smoothing_length_generation.generate_smoothing_lengths`.

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.projection import project_pixel_grid
   from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

   data = load("my_snapshot_0000.hdf5")

   # Generate smoothing lengths for the dark matter
   data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
       data.dark_matter.coordinates,
       data.metadata.boxsize,
       kernel_gamma=1.8,
       neighbours=57,
       speedup_fac=2,
       dimension=3,
   )

   # Project the dark matter mass
   dm_mass = project_pixel_grid(
       # Note here that we pass in the dark matter dataset not the whole
       # data object, to specify what particle type we wish to visualise
       data=data.dark_matter,
       boxsize=data.metadata.boxsize,
       resolution=1024,
       project="masses",
       parallel=True,
       region=None
   )

   from matplotlib.pyplot import imsave
   from matplotlib.colors import LogNorm

   # Everyone knows that dark matter is purple
   imsave("dm_mass_map.png", LogNorm()(dm_mass), cmap="inferno")


Lower-level API
---------------

The lower-level API for projections allows for any general positions,
smoothing lengths, and smoothed quantities, to generate a pixel grid that
represents the smoothed version of the data.

This API is available through
:meth:`swiftsimio.visualisation.projection.scatter` and
:meth:`swiftsimio.visualisation.projection.scatter_parallel` for the parallel
version. The parallel version uses significantly more memory as it allocates
a thread-local image array for each thread, summing them in the end. Here we
will only describe the ``scatter`` variant, but they behave in the exact same way.

To use this function, you will need:

+ x-positions of all of your particles, ``x``.
+ y-positions of all of your particles, ``y``.
+ A quantity which you wish to smooth for all particles, such as their
  mass, ``m``.
+ Smoothing lengths for all particles, ``h``.
+ The resolution you wish to make your square image at, ``res``.

The key here is that only particles in the domain [0, 1] in x, and [0, 1] in y
will be visible in the image. You may have particles outside of this range;
they will not crash the code, and may even contribute to the image if their
smoothing lengths overlap with [0, 1]. You will need to re-scale your data
such that it lives within this range. Then you may use the function as follows:

.. code-block:: python

   from swiftsimio.visualisation.projection import scatter

   # Using the variable names from above
   out = scatter(x=x, y=y, h=h, m=m, res=res)

``out`` will be a 2D :mod:`numpy` grid of shape ``[res, res]``. You will need
to re-scale this back to your original dimensions to get it in the correct units,
and do not forget that it now represents the smoothed quantity per surface area.
