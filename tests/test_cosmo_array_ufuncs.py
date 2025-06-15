"""
Tests that ufuncs handling cosmo_array's properly handle our extra attributes.
"""

import pytest
import numpy as np
import unyt as u
from swiftsimio.objects import (
    cosmo_array,
    cosmo_quantity,
    cosmo_factor,
    a,
    multiple_output_operators,
    InvalidScaleFactor,
)


class TestCopyFuncs:
    """
    Test ufuncs that copy arrays.
    """

    @pytest.mark.parametrize(
        ("func"), ["byteswap", "diagonal", "flatten", "ravel", "transpose", "view"]
    )
    def test_argless_copyfuncs(self, func):
        """
        Make sure that our attributes are preserved through copying functions that
        take no arguments.
        """
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        assert hasattr(getattr(arr, func)(), "cosmo_factor")
        assert hasattr(getattr(arr, func)(), "comoving")

    def test_astype(self):
        """
        Make sure that our attributes are preserved through astype.
        """
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
        """
        Make sure that our attributes are preserved through in_units (from unyt).
        """
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
        """
        Make sure that our attributes are preserved through compress.
        """
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
        """
        Make sure that our attributes are preserved through repeat.
        """
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
        """
        Make sure that our attributes are preserved through transpose (T).
        """
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
        """
        Make sure that our attributes are preserved through ua (from unyt).
        """
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
        """
        Make sure that our attributes are preserved through unit_array (from unyt).
        """
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor("a^1", 1),
            comoving=False,
        )
        res = arr.unit_array
        assert hasattr(res, "cosmo_factor")
        assert hasattr(res, "comoving")

    def test_compatibility(self):
        """
        Check that the compatible_with_comoving and compatible_with_physical functions
        give correct compatibility checks.
        """
        # comoving array at high redshift
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 1, 0.5),
            comoving=True,
        )
        assert arr.compatible_with_comoving()
        assert not arr.compatible_with_physical()
        # physical array at high redshift
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 1, 0.5),
            comoving=False,
        )
        assert not arr.compatible_with_comoving()
        assert arr.compatible_with_physical()
        # comoving array with no scale factor dependency at high redshift
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 0, 0.5),
            comoving=True,
        )
        assert arr.compatible_with_comoving()
        assert arr.compatible_with_physical()
        # physical array with no scale factor dependency at high redshift
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 0, 0.5),
            comoving=False,
        )
        assert arr.compatible_with_comoving()
        assert arr.compatible_with_physical()
        # comoving array at redshift 0
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 1, 1.0),
            comoving=True,
        )
        assert arr.compatible_with_comoving()
        assert arr.compatible_with_physical()
        # physical array at redshift 0
        arr = cosmo_array(
            np.ones((10, 10)),
            units="Mpc",
            cosmo_factor=cosmo_factor(a ** 1, 1.0),
            comoving=False,
        )
        assert arr.compatible_with_comoving()
        assert arr.compatible_with_physical()


class TestCheckUfuncCoverage:
    """
    Check that we've wrapped all functions that unyt wraps.
    """

    def test_multi_output_coverage(self):
        """
        Compare our list of multi_output_operators with unyt's to make sure we cover
        everything.
        """
        assert set(multiple_output_operators.keys()) == set(
            (np.modf, np.frexp, np.divmod)
        )

    def test_ufunc_coverage(self):
        """
        Compare our list of ufuncs with unyt's to make sure we cover everything.
        """
        assert set(u.unyt_array._ufunc_registry.keys()) == set(
            cosmo_array._ufunc_registry.keys()
        )


class TestCosmoArrayUfuncs:
    """
    Test some example functions using each of our wrappers for correct output.
    """

    def test_preserving_ufunc(self):
        """
        Tests of the _preserve_cosmo_factor wrapper.
        """
        # 1 argument
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.ones_like(inp)
        assert res.to_value(u.kpc) == 1
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor
        # 2 argument, no cosmo_factors
        inp = cosmo_array([2], u.kpc, comoving=False)
        res = inp + inp
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(None, None)
        # 2 argument, one is not cosmo_array
        inp1 = u.unyt_array([2], u.kpc)
        inp2 = cosmo_array([2], u.kpc, comoving=False)
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            res = inp1 + inp2
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(None, None)
        # 2 argument, two is not cosmo_array
        inp1 = cosmo_array([2], u.kpc, comoving=False)
        inp2 = u.unyt_array([2], u.kpc)
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            res = inp1 + inp2
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(None, None)
        # 2 argument, only one has cosmo_factor
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array([2], u.kpc, comoving=False)
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            inp1 + inp2
        # 2 argument, only two has cosmo_factor
        inp1 = cosmo_array([2], u.kpc, comoving=False)
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            inp1 + inp2
        # 2 argument, mismatched cosmo_factors
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            inp1 + inp2
        # 2 argument, matched cosmo_factors
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp + inp
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor

    def test_multiplying_ufunc(self):
        """
        Tests of the _multiply_cosmo_factor wrapper.
        """
        # no cosmo_factors
        inp = cosmo_array([2], u.kpc, comoving=False)
        res = inp * inp
        assert res.to_value(u.kpc ** 2) == 4
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(None, None)
        # one is not cosmo_array
        inp1 = 2
        inp2 = cosmo_array([2], u.kpc, comoving=False)
        res = inp1 * inp2
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == inp2.cosmo_factor
        # two is not cosmo_array
        inp1 = cosmo_array([2], u.kpc, comoving=False)
        inp2 = 2
        res = inp1 * inp2
        assert res.to_value(u.kpc) == 4
        assert res.comoving is False
        assert res.cosmo_factor == inp1.cosmo_factor
        # only one has cosmo_factor
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array([2], u.kpc, comoving=False)
        with pytest.raises(InvalidScaleFactor, match="Attempting to multiply"):
            inp1 * inp2
        # only two has cosmo_factor
        inp1 = cosmo_array([2], u.kpc, comoving=False)
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        with pytest.raises(InvalidScaleFactor, match="Attempting to multiply"):
            inp1 * inp2
        # cosmo_factors both present
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp * inp
        assert res.to_value(u.kpc ** 2) == 4
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** 2

    def test_dividing_ufunc(self):
        """
        Tests of the _divide_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [2.0],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp / inp
        assert res.to_value(u.dimensionless) == 1  # also ensures units ok
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** 0

    def test_return_without_ufunc(self):
        """
        Tests of the _return_without_cosmo_factor wrapper.
        """
        # 1 argument
        inp = cosmo_array(
            [1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.logical_not(inp)
        assert res == np.logical_not(1)
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)
        # 2 arguments, both cosmo_array
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.logaddexp(inp, inp)
        assert res == np.logaddexp(2, 2)
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)
        # 2 arguments, mismatched cosmo_factor
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            np.logaddexp(inp1, inp2)
        # 2 arguments, one missing comso_factor
        inp1 = cosmo_array([2], u.kpc, comoving=False)
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            np.logaddexp(inp1, inp2)
        # 2 arguments, two missing comso_factor
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array([2], u.kpc, comoving=False)
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            np.logaddexp(inp1, inp2)
        assert res == np.logaddexp(2, 2)
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)
        # 2 arguments, one not cosmo_array
        inp1 = u.unyt_array([2], u.kpc)
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            res = np.logaddexp(inp1, inp2)
        assert res == np.logaddexp(2, 2)
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)
        # 2 arguments, two not cosmo_array
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = u.unyt_array([2], u.kpc)
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            res = np.logaddexp(inp1, inp2)
        assert res == np.logaddexp(2, 2)
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)

    def test_sqrt_ufunc(self):
        """
        Tests of the _sqrt_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [4],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.sqrt(inp)
        assert res.to_value(u.kpc ** 0.5) == 2  # also ensures units ok
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** 0.5

    def test_square_ufunc(self):
        """
        Tests of the _square_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.square(inp)
        assert res.to_value(u.kpc ** 2) == 4  # also ensures units ok
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** 2

    def test_cbrt_ufunc(self):
        """
        Tests of the _cbrt_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [8],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.cbrt(inp)
        assert res.to_value(u.kpc ** (1.0 / 3.0)) == 2  # also ensures units ok
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** (1.0 / 3.0)

    def test_reciprocal_ufunc(self):
        """
        Tests of the _reciprocal_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [2.0],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.reciprocal(inp)
        assert res.to_value(u.kpc ** -1) == 0.5  # also ensures units ok
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** -1

    def test_passthrough_ufunc(self):
        """
        Tests of the _passthrough_cosmo_factor wrapper.
        """
        # 1 argument
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.negative(inp)
        assert res.to_value(u.kpc) == -2
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor
        # 2 argument, matching
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.copysign(inp, inp)
        assert res.to_value(u.kpc) == inp.to_value(u.kpc)
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor
        # 2 argument, not matching
        inp1 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        with pytest.raises(
            ValueError, match="Arguments have cosmo_factors that differ"
        ):
            np.copysign(inp1, inp2)

    def test_arctan2_ufunc(self):
        """
        Tests of the _arctan2_cosmo_factor wrapper.
        """
        inp = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.arctan2(inp, inp)
        assert res.to_value(u.dimensionless) == np.arctan2(2, 2)
        assert res.comoving is False
        assert res.cosmo_factor.a_factor == 1  # also ensures cosmo_factor present

    def test_comparison_ufunc(self):
        """
        Tests of the _comparison_cosmo_factor wrapper.
        """
        inp1 = cosmo_array(
            [1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array(
            [2],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp1 < inp2
        assert res.all()
        assert isinstance(res, np.ndarray) and not isinstance(res, u.unyt_array)

    def test_out_arg(self):
        """
        Test that our helpers can handle functions with an ``out`` kwarg.
        """
        inp = cosmo_array(
            [1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        out = cosmo_array([np.nan], u.dimensionless, comoving=True)
        np.abs(inp, out=out)
        assert out.to_value(u.kpc) == np.abs(inp.to_value(u.kpc))
        assert out.comoving is False
        assert out.cosmo_factor == inp.cosmo_factor
        inp = cosmo_array(
            [1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        # make sure we can also pass a non-cosmo type for out without crashing
        out = np.array([np.nan])
        np.abs(inp, out=out)
        assert out == np.abs(inp.to_value(u.kpc))

    def test_reduce_multiply(self):
        """
        Test that we can handle the reduce method for the multiply ufunc.
        """
        inp = cosmo_array(
            [[1, 2], [3, 4]],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        res = np.multiply.reduce(inp, axis=0)
        np.testing.assert_allclose(res.to_value(u.kpc ** 2), np.array([3.0, 8.0]))
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** 2

    def test_reduce_divide(self):
        """
        Test that we can handle the reduce method for the divide ufunc.
        """
        inp = cosmo_array(
            [[1.0, 2.0], [1.0, 4.0], [1.0, 1.0]],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        res = np.divide.reduce(inp, axis=0)
        np.testing.assert_allclose(res.to_value(u.kpc ** -1), np.array([1.0, 0.5]))
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor ** -1

    def test_reduce_other(self):
        """
        Test that we can handle other ufuncs with a reduce method.
        """
        inp = cosmo_array(
            [[1.0, 2.0], [1.0, 2.0]],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = np.add.reduce(inp, axis=0)
        np.testing.assert_allclose(res.to_value(u.kpc), np.array([2.0, 4.0]))
        assert res.comoving is False
        assert res.cosmo_factor == inp.cosmo_factor

    def test_multi_output(self):
        """
        Test that we can handle functions with multiple return values.
        """
        # with passthrough
        inp = cosmo_array(
            [2.5],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res1, res2 = np.modf(inp)
        assert res1.to_value(u.kpc) == 0.5
        assert res2.to_value(u.kpc) == 2.0
        assert res1.comoving is False
        assert res2.comoving is False
        assert res1.cosmo_factor == inp.cosmo_factor
        assert res2.cosmo_factor == inp.cosmo_factor
        # with return_without
        inp = cosmo_array(
            [2.5],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res1, res2 = np.frexp(inp)
        assert res1 == 0.625
        assert res2 == 2
        assert isinstance(res1, np.ndarray) and not isinstance(res1, u.unyt_array)
        assert isinstance(res2, np.ndarray) and not isinstance(res2, u.unyt_array)

    def test_multi_output_with_out_arg(self):
        """
        Test that we can handle multiple return values in conjunction with an ``out``
        kwarg.
        """
        # with two out arrays
        inp = cosmo_array(
            [2.5],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        out1 = cosmo_array([np.nan], u.dimensionless, comoving=True)
        out2 = cosmo_array([np.nan], u.dimensionless, comoving=True)
        np.modf(inp, out=(out1, out2))
        assert out1.to_value(u.kpc) == 0.5
        assert out2.to_value(u.kpc) == 2.0
        assert out1.comoving is False
        assert out2.comoving is False
        assert out1.cosmo_factor == inp.cosmo_factor
        assert out2.cosmo_factor == inp.cosmo_factor

    def test_comparison_with_zero(self):
        """
        Test that we don't produce warnings for dangerous comparisons on comparison with
        zero.
        """
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = 0
        res = inp1 > inp2
        assert res.all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = 0.5
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            res = inp1 > inp2
        assert res.all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_array(
            [0, 0, 0],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        assert (inp1 > inp2).all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = np.ones(3) * u.kpc
        with pytest.warns(RuntimeWarning, match="Mixing arguments"):
            assert (inp1 == inp2).all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = np.zeros(3) * u.kpc
        assert (inp1 > inp2).all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_quantity(
            1,
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp1 == inp2
        assert res.all()
        inp1 = cosmo_array(
            [1, 1, 1],
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        inp2 = cosmo_quantity(
            0,
            u.kpc,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=1.0),
        )
        res = inp1 > inp2
        assert res.all()


class TestComovingConversion:
    def test_conversion_happens(self):
        """
        Given a physical and a comoving input to e.g. addition, conversion
        should happen and we should get a correct result.
        """
        inp1 = cosmo_array(
            [1, 2, 3],
            u.kpc,
            comoving=True,
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),
        )
        inp2 = cosmo_array(
            [1, 2, 3],
            u.Mpc,  # different units, expect conversion
            comoving=False,  # different comoving, expect conversion
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=0.5),  # not z=0
        )
        result = inp1 + inp2
        assert np.allclose(
            result.to_comoving_value(u.kpc), np.array([2001, 4002, 6003])
        )
