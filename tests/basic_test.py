"""
Basic integration test!
"""

from swiftsimio import load
from swiftsimio.writer import SWIFTWriterDataset
from swiftsimio.units import cosmo_units

import unyt
import numpy as np

import os

def test_write():
    """
    Creates a sample dataset.
    """
    x = SWIFTWriterDataset(cosmo_units, 100)

    n_p = 32**3

    x.gas.coordinates = unyt.unyt_array(100*np.random.rand(n_p*3).reshape(n_p, 3), unyt.kpc*1000)

    x.gas.velocities = unyt.unyt_array(np.random.rand(n_p*3).reshape(n_p, 3), unyt.km / unyt.s)

    x.gas.smoothing_length = unyt.unyt_array(np.ones(n_p, dtype=float) * 100 / (n_p**(1/3)), unyt.kpc*1000)

    x.gas.masses = unyt.unyt_array(np.ones(n_p, dtype=float)*1e6, unyt.msun)

    x.gas.internal_energy = unyt.unyt_array(np.ones(n_p, dtype=float)*1e-19, unyt.J)
    
    x.write("test.hdf5")


def test_load():
    """
    Tries to load the dataset we just made.
    """
    x = load("test.hdf5")

    density = x.gas.internal_energy
    coordinates = x.gas.coordinates

    os.remove("test.hdf5")    