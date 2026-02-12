Creating Initial Conditions
===========================

Writing datasets that are valid for consumption for cosmological codes can be
difficult, especially when considering how to best use units. SWIFT uses a
different set of internal units (specified in your parameter file) that does
not necessarily need to be the same set of units that initial conditions are
specified in. Nevertheless, it is important to ensure that units in the
initial conditions are all *consistent* with each other. To facilitate this,
we use :class:`~swiftsimio.objects.cosmo_array` data. The below example
generates randomly placed gas particles with uniform densities.

The functionality to create initial conditions is available through
the :mod:`swiftsimio.snapshot_writer` sub-module, and the top-level
:class:`~swiftsimio.Writer` class.

.. warning::

   The properties that :mod:`swiftsimio` requires in the initial
   conditions are the only ones that are actually read by SWIFT; other fields
   will be left un-read and as such cannot be included in initial conditions
   files using the :class:`~swiftsimio.Writer`. Any additional
   attributes set will be silently ignored.

.. warning::

   You need to be careful that your choice of unit system does
   *not* allow values over 2^31, i.e. you need to ensure that your
   provided values (with units) when *written* to the file are safe to 
   be interpreted as (single-precision) floats. The only exception to
   this is coordinates which are stored in double precision.

Example
^^^^^^^

.. code-block:: python

   import numpy as np
   import unyt as u
   from swiftsimio import Writer, cosmo_array
   from swiftsimio.metadata.writer.unit_systems import cosmo_unit_system

   # number of gas particles
   n_p = 1000
   # scale factor of 1.0
   a = 1.0
   # Box is 100 Mpc
   lbox = 100
   boxsize = cosmo_array(
        [lbox, lbox, lbox],
        u.Mpc,
        comoving=True,
        scale_factor=a,
        scale_exponent=1,
   )
   
   # Create the Writer object. cosmo_unit_system corresponds to default Gadget-like units
   # of 10^10 Msun, Mpc, and km/s
   w = Writer(unit_system=cosmo_unit_system, boxsize=boxsize, scale_factor=a)

   # Randomly spaced coordinates from 0 to lbox Mpc in each direction
   w.gas.coordinates = cosmo_array(
       np.random.rand(n_p, 3) * lbox,
       u.Mpc,
       comoving=True,
       scale_factor=w.scale_factor,
       scale_exponent=1,
   )

   # Random velocities from 0 to 1 km/s
   w.gas.velocities = cosmo_array(
       np.random.rand(n_p, 3),
       u.km / u.s,
       comoving=True,
       scale_factor=w.scale_factor,
       scale_exponent=1,
   )

   # Generate uniform masses as 10^6 solar masses for each particle
   w.gas.masses = cosmo_array(
       np.ones(n_p, dtype=float) * 1e6,
       u.msun,
       comoving=True,
       scale_factor=w.scale_factor,
       scale_exponent=0,
   )

   # Generate internal energy corresponding to 10^4 K
   w.gas.internal_energy = cosmo_array(
       np.ones(n_p, dtype=float) * 1e4 / 1e6,
       u.kb * u.K / u.solMass,
       comoving=True,
       scale_factor=w.scale_factor,
       scale_exponent=-2,
   )

   # Generate initial guess for smoothing lengths based on mean inter-particle spacing
   w.gas.generate_smoothing_lengths()

   # w.gas.particle_ids can optionally be set, otherwise they are auto-generated

   # write the initial conditions out to a file
   w.write("ics.hdf5")

Then, running ``h5glance`` (``pip install h5glance``) on the resulting ``ics.hdf5``
produces:

.. code-block:: bash

   ics.hdf5
   ├Header (9 attributes)
   ├PartType0
   │ ├Coordinates	[float64: 1000 × 3] (9 attributes)
   │ ├InternalEnergy	[float64: 1000] (9 attributes)
   │ ├Masses	[float64: 1000] (9 attributes)
   │ ├ParticleIDs	[int64: 1000] (9 attributes)
   │ ├SmoothingLengths	[float64: 1000] (9 attributes)
   │ └Velocities	[float64: 1000 × 3] (9 attributes)
   └Units (5 attributes)
