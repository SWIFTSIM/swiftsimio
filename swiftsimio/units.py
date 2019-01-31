"""
Contains unit systems that may be useful to astronomers. In particular,
it contains the cosmo_units which can be considered Gadget-oid default units,
with

+ Unit length = Mpc
+ Unit velocity = km/s
+ Unit mass = 10^10 Msun
+ Unit temperature = K
"""

import unyt

cosmo_units = unyt.UnitSystem(
    "cosmological",
    unyt.Mpc,
    1e10 * unyt.msun,
    unyt.s * unyt.Mpc / unyt.km
)
