"""
Tests the initialisation of a cosmo_array.
"""

import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array, cosmo_factor


class TestCosmoArrayInit:

    def test_init_from_ndarray(self):
        arr = cosmo_array(
            np.ones(5),
            units=u.Mpc,
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list(self):
        arr = cosmo_array(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_unyt_array(self):
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=u.Mpc),
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_unyt_arrays(self):
        arr = cosmo_array(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)
