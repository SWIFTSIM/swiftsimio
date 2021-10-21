"""
Tests that functions returning copies of cosmo_array
 preserve the comoving and cosmo_factor attributes.
"""

import pytest
import numpy as np
from swiftsimio.objects import cosmo_array, cosmo_factor


class TestCopyFuncs:
    @pytest.mark.parametrize(
        ("func"),
        [
            "byteswap",
            "diagonal",
            "flatten",
            "newbyteorder",
            "ravel",
            "transpose",
            "view",
        ],
    )
    def test_argless_copyfuncs(self, func):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        assert hasattr(getattr(arr, func)(), "cosmo_factor")
        assert hasattr(getattr(arr, func)(), "comoving")

    def test_astype(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.astype(np.float64)
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_in_units(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.in_units("kpc")
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_compress(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.compress([True])
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_repeat(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.repeat(2)
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_T(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.T
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_ua(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.ua
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_unit_array(self):
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.unit_array
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")
