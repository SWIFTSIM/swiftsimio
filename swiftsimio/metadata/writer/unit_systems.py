"""
Define unit systems that may be useful to astronomers.

In particular, we define ``cosmo_units`` that are Gadget-like default units:

+ Unit length = Mpc
+ Unit velocity = km/s
+ Unit mass = 10^10 Msun
+ Unit temperature = K

Also contains unit conversion factors, to simplify units wherever possible.
"""

import unyt

cosmo_units = unyt.UnitSystem(
    "cosmological",
    unyt.Mpc,
    unyt.unyt_quantity(1e10, units=unyt.solMass),
    unyt.unyt_quantity(1.0, units=unyt.s * unyt.Mpc / unyt.km).to(unyt.Gyr),
)
