"""
Contains unit systems that may be useful to astronomers. In particular,
it contains the cosmo_units which can be considered Gadget-oid default units,
with.

+ Unit length = Mpc
+ Unit velocity = km/s
+ Unit mass = 10^10 Msun
+ Unit temperature = K

Also contains unit conversion factors, to simplify units wherever possible.
"""

import unyt


try:
    # Need to do this first otherwise the `unyt` system freaks out about
    # us upgrading msun from a symbol
    cosmo_units = unyt.UnitSystem(
        "cosmological",
        unyt.Mpc,
        unyt.unyt_quantity(1e10, units=unyt.Solar_Mass),
        unyt.unyt_quantity(1.0, units=unyt.s * unyt.Mpc / unyt.km).to(unyt.Gyr),
    )
except RuntimeError:
    # We've already done that, oops.
    cosmo_units = unyt.unit_systems.cosmological
    pass
