"""
Contains the names and types of the units as read from the file.
"""

import unyt
from numpy import log

unit_names_to_unyt = {
    "Unit mass in cgs (U_M)": unyt.g,
    "Unit length in cgs (U_L)": unyt.cm,
    "Unit time in cgs (U_t)": unyt.s,
    "Unit current in cgs (U_I)": unyt.statA,
    "Unit temperature in cgs (U_T)": unyt.K,
}


possible_base_units = {
    "mass": [unyt.g, unyt.kg, unyt.Mearth, unyt.Solar_Mass],
    "length": [
        unyt.cm,
        unyt.m,
        unyt.km,
        unyt.Rearth,
        unyt.pc,
        unyt.kpc,
        unyt.Mpc,
        unyt.Gpc,
    ],
    "time": [unyt.s, unyt.year, unyt.Myr, unyt.Gyr],
    "current": [unyt.statA],
    "temperature": [unyt.K],
}


def find_nearest_base_unit(unit: unyt.unyt_quantity, dimension: str):
    """
    Uses the possible_base_units dictionary to find the closest
    base unit to your internal units, and returns that.

    Example
    -------

    .. code-block::python

        find_nearest_base_unit(1e43 * unyt.g, "mass")
        >>> unyt.Solar_Mass
    """

    possible_bases = possible_base_units[dimension]

    return unit.to(min(possible_bases, key=lambda x: abs(log((1.0 * x).to(unit)))))
