Command-line Utilities
======================

:mod:`swiftsimio` comes with some useful command-line utilities.
Basic documentation for these is provided below, but you can always
find up-to-date documentation by invoking these with ``-h`` or
``--help``.

``swiftsnap``
-------------

The ``swiftsnap`` utility, introduced in :mod:`swiftsimio` version
3.1.2, allows you to preview the metadata inside a SWIFT snapshot
file. Simply invoke it with the path to a snapshot, and it will
show you a selection of useful metadata. See below for an example.

.. code:: bash

    swiftsnap output_0103.hdf5

Produces the following output:

    Untitled SWIFT simulation
    Written at: 2020-06-01 08:44:51
    Active policies: cosmological integration, hydro, keep, self gravity, steal
    Output type: Snapshot, Output selection: Snapshot
    LLVM/Clang (11.0.0)
    Non-MPI version of SWIFT
    SWIFT (io_selection_changes)
    v0.8.5-725-g10d7d5b3-dirty
    2020-05-29 18:00:58 +0100
    Simulation state: z=0.8889, a=0.5294, t=6.421 Gyr
    H_0=70.3 km/(Mpc*s), ρ_crit=1.433e-05 cm**(-3)
    Ω_b=0.0455, Ω_k=0, Ω_lambda=0.724, Ω_m=0.276, Ω_r=0
    ω=-1, ω_0=-1, ω_a=0
    Gravity scheme: With per-particle softening
    Hydrodynamics scheme: Gadget-2 version of SPH (Springel 2005)
    Chemistry model: None
    Cooling model: None
    Entropy floor: None
    Feedback model: None
    Tracers: None

