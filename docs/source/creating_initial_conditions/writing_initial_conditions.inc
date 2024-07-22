Writing Initial Conditions
--------------------------

Writing datasets that are valid for consumption for cosmological codes can be
difficult, especially when considering how to best use units. SWIFT uses a
different set of internal units (specified in your parameter file) that does
not necessarily need to be the same set of units that initial conditions are
specified in. Nevertheless, it is important to ensure that units in the
initial conditions are all *consistent* with each other. To facilitate this,
we use :mod:`unyt` arrays. The below example generates randomly placed gas
particles with uniform densities.

The functionality to create initial conditions is available through
the :mod:`swiftsimio.writer` sub-module, and the top-level
:obj:`swiftsimio.Writer` object.

Note that the properties that :mod:`swiftsimio` requires in the initial
conditions are the only ones that are actually read by SWIFT; other fields
will be left un-read and as such should not be included in initial conditions
files.

A current known issue is that due to inconsistencies with the initial
conditions and simulation snapshots, :mod:`swiftsimio` is not actually able
to read the inititial conditions that it produces. We are aiming to fix this
in an upcoming release.


Example
^^^^^^^

.. code-block:: python

   from swiftsimio import Writer
   from swiftsimio.units import cosmo_units

   import unyt
   import numpy as np

   # Box is 100 Mpc
   boxsize = 100 * unyt.Mpc

   # Generate object. cosmo_units corresponds to default Gadget-oid units
   # of 10^10 Msun, Mpc, and km/s
   x = Writer(cosmo_units, boxsize)

   # 32^3 particles.
   n_p = 32**3

   # Randomly spaced coordinates from 0, 100 Mpc in each direction
   x.gas.coordinates = np.random.rand(n_p, 3) * (100 * unyt.Mpc)

   # Random velocities from 0 to 1 km/s
   x.gas.velocities = np.random.rand(n_p, 3) * (unyt.km / unyt.s)

   # Generate uniform masses as 10^6 solar masses for each particle
   x.gas.masses = np.ones(n_p, dtype=float) * (1e6 * unyt.msun)

   # Generate internal energy corresponding to 10^4 K
   x.gas.internal_energy = (
       np.ones(n_p, dtype=float) * (1e4 * unyt.kb * unyt.K) / (1e6 * unyt.msun)
   )

   # Generate initial guess for smoothing lengths based on MIPS
   x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

   # If IDs are not present, this automatically generates
   x.write("test.hdf5")

Then, running ``h5glance`` on the resulting ``test.hdf5`` produces:

.. code-block:: bash

   test.hdf5
   ├Header
   │ └5 attributes:
   │   ├BoxSize: 100.0
   │   ├Dimension: array [int64: 1]
   │   ├Flag_Entropy_ICs: 0
   │   ├NumPart_Total: array [int64: 6]
   │   └NumPart_Total_HighWord: array [int64: 6]
   ├PartType0
   │ ├Coordinates  [float64: 32768 × 3]
   │ ├InternalEnergy       [float64: 32768]
   │ ├Masses       [float64: 32768]
   │ ├ParticleIDs  [float64: 32768]
   │ ├SmoothingLength      [float64: 32768]
   │ └Velocities   [float64: 32768 × 3]
   └Units
   └5 attributes:
       ├Unit current in cgs (U_I): array [float64: 1]
       ├Unit length in cgs (U_L): array [float64: 1]
       ├Unit mass in cgs (U_M): array [float64: 1]
       ├Unit temperature in cgs (U_T): array [float64: 1]
       └Unit time in cgs (U_t): array [float64: 1]

**Note** you do need to be careful that your choice of unit system does
*not* allow values over 2^31, i.e. you need to ensure that your
provided values (with units) when *written* to the file are safe to 
be interpreted as (single-precision) floats. The only exception to
this is coordinates which are stored in double precision.
