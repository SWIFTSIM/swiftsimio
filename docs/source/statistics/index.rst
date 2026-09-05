Statistics Files
================

:mod:`swiftsimio` includes routines to load log files, such as the
``SFR.txt`` and ``energy.txt``. This is available through the
:obj:`swiftsimio.statistics.SWIFTStatisticsFile` object, or through
the main ``load_statistics`` function.

Column headings are normalized to valid Python identifiers for attribute access. For
example, a column named ``SFR (total)`` is available as ``data.sfr_total``.

Example
-------

.. code-block:: python

   from swiftsimio import load_statistics

   data = load_statistics("energy.txt")

   print(data)

   print(x.total_mass.name)


Will output:

.. code-block:: bash

   Statistics file: energy.txt, containing fields: _var, step, time, a, z, total_mass,
   gas_mass, dm_mass, sink_mass, star_mass, bh_mass, gas_z_mass, star_z_mass,
   bh_z_mass, kin_energy, int_energy, pot_energy, rad_energy, gas_entropy, com_x,
   com_y, com_z, mom_x, mom_y, mom_z, ang_mom_x, ang_mom_y, ang_mom_z

   'Total mass in the simulation'
