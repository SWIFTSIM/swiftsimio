"""
Contains the names and types of the units as read from the file.
"""

from unyt import g, cm, s, statA, K

unit_names_to_unyt = {
    "Unit mass in cgs (U_M)": g,
    "Unit length in cgs (U_L)": cm,
    "Unit time in cgs (U_t)": s,
    "Unit current in cgs (U_I)": statA,
    "Unit temperature in cgs (U_T)": K
}

