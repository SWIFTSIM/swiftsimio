"""Basic integration test."""

import os
import pytest
import numpy as np
from swiftsimio import load, cosmo_array, Writer
import unyt as u


def test_write(simple_snapshot_data):
    """Create a sample dataset. Should not crash."""
    _, testfile = simple_snapshot_data
    assert os.path.isfile(testfile)


def test_load(simple_snapshot_data):
    """Try to load a dataset made by the writer. Should not crash."""
    _, testfile = simple_snapshot_data
    dat = load(testfile)
    dat.gas.internal_energy
    dat.gas.coordinates


def test_write_non_required_field(simple_writer):
    """
    Try to write a non-required field with the writer.

    Expectation is that it is silently ignored and not written out. Written file should
    still be readable.
    """
    testfile = "write_non_required_field.hdf5"
    simple_writer.gas.metallicity = cosmo_array(
        np.ones(simple_writer.gas.masses.size),
        u.dimensionless,
        comoving=True,
        scale_factor=simple_writer.scale_factor,
        scale_exponent=0.0,
    )
    try:
        simple_writer.write(testfile)
        dat = load(testfile)
        with pytest.raises(
            AttributeError, match="'GasDataset' object has no attribute 'metallicity'"
        ):
            dat.gas.metallicity
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)


def test_write_missing_required(simple_writer):
    """
    Try to write with a required field missing.

    Expectation is to raise with a helpful message.
    """
    testfile = "write_missing_required.hdf5"
    del simple_writer.gas.coordinates
    try:
        with pytest.raises(
            AttributeError, match="Required dataset coordinates is None."
        ):
            simple_writer.write(testfile)
    finally:
        # we don't expect to write the file but cleanup in case the unexpected happens
        if os.path.exists(testfile):
            os.remove(testfile)


class TestSetterInputs:
    """Tests for the setter methods with valid and invalid inputs."""

    def test_setter_valid_input(self):
        """Make sure setter accepts valid input and performs expected conversions."""
        a = 0.5
        w = Writer(
            boxsize=cosmo_array(
                [100, 100, 100], u.Mpc, comoving=True, scale_factor=a, scale_exponent=1
            ),
            scale_factor=a,
        )
        input_units = u.kpc
        input_values = np.arange(300, dtype=float).reshape((100, 3))
        expected_units = w.unit_system[u.dimensions.length]
        # a valid input, should be accepted:
        w.gas.coordinates = cosmo_array(
            input_values.copy(),  # copy else we modify through this view
            input_units,
            comoving=False,
            scale_factor=a,
            scale_exponent=1.0,
        )
        # check that we converted units
        assert input_units != expected_units
        assert np.allclose(
            w.gas.coordinates.to_physical_value(input_units), input_values
        )
        # check that we converted to comoving
        assert w.gas.coordinates.cosmo_factor.a_factor != 1.0  # else this is trivial
        assert w.gas.coordinates.comoving
        assert np.allclose(
            w.gas.coordinates.to_comoving_value(input_units),
            input_values / w.gas.coordinates.cosmo_factor.a_factor,
        )

    def test_setter_invalid_shape_input(self, simple_writer):
        """
        Try to set a dataset with a transposed coordinate array.

        Should be rejected on attempted write.
        """
        del simple_writer.gas.coordinates
        n_p = simple_writer.gas.masses.size
        simple_writer.gas.coordinates = cosmo_array(
            np.arange(3 * n_p, dtype=float).reshape(
                (3, n_p)
            ),  # (n_p, 3) is valid shape
            u.Mpc,
            comoving=True,
            scale_factor=simple_writer.scale_factor,
            scale_exponent=1.0,
        )
        testfile = "setter_invalid_coordinate_shape_input.hdf5"
        try:
            with pytest.raises(
                AssertionError,
                match="Arrays passed to gas dataset are not of the same size.",
            ):
                simple_writer.write(testfile)
        finally:
            # expect not to write a file but clean up if the unexpected happens
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_setter_invalid_dimensions_input(self):
        """Make sure setter refuses input with invalid units."""
        a = 0.5
        w = Writer(
            boxsize=cosmo_array(
                [100, 100, 100], u.Mpc, comoving=True, scale_factor=a, scale_exponent=1
            ),
            scale_factor=a,
        )
        testfile = "setter_invalid_dimensions_input.hdf5"
        try:
            with pytest.raises(
                u.exceptions.InvalidUnitEquivalence, match="The unit equivalence"
            ):
                w.gas.masses = cosmo_array(
                    np.arange(100),
                    u.Mpc,
                    comoving=True,
                    scale_factor=a,
                    scale_exponent=1.0,
                )
        finally:
            # expect not to write a file but clean up if the unexpected happens
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_setter_invalid_scale_factor_input(self):
        """Make sure setter refuses input with mismatched scale factor."""
        a = 0.5
        w = Writer(
            boxsize=cosmo_array(
                [100, 100, 100], u.Mpc, comoving=True, scale_factor=a, scale_exponent=1
            ),
            scale_factor=a,
        )
        testfile = "setter_invalid_scale_factor_input.hdf5"
        a_input = a + 0.1
        assert a_input != a
        try:
            with pytest.raises(
                AssertionError,
                match="The scale factor of masses does not match the scale factor of the "
                "Writer.",
            ):
                w.gas.masses = cosmo_array(
                    np.arange(100),
                    u.solMass,
                    comoving=True,
                    scale_factor=a_input,
                    scale_exponent=1.0,
                )
        finally:
            # expect not to write a file but clean up if the unexpected happens
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_setter_unyt_array_input(self):
        """Make sure setter refuses unyt_array (not cosmo_array) input."""
        a = 0.5
        w = Writer(
            boxsize=cosmo_array(
                [100, 100, 100], u.Mpc, comoving=True, scale_factor=a, scale_exponent=1
            ),
            scale_factor=a,
        )
        testfile = "setter_unyt_array_input.hdf5"
        try:
            with pytest.raises(
                TypeError, match="Provide masses as swiftsimio.cosmo_array."
            ):
                w.gas.masses = np.arange(100) * u.solMass
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_setter_ndarray_input(self):
        """Make sure setter refuses numpy array (not cosmo_array) input."""
        a = 0.5
        w = Writer(
            boxsize=cosmo_array(
                [100, 100, 100], u.Mpc, comoving=True, scale_factor=a, scale_exponent=1
            ),
            scale_factor=a,
        )
        testfile = "setter_ndarray_input.hdf5"
        try:
            with pytest.raises(
                TypeError, match="Provide masses as swiftsimio.cosmo_array."
            ):
                w.gas.masses = np.arange(100)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)

    @pytest.mark.parametrize(
        "input_data",
        (
            np.arange(100, dtype=int),
            np.arange(100, dtype=int) * u.dimensionless,
            cosmo_array(
                np.arange(100, dtype=int),
                u.dimensionless,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
        ),
    )
    def test_setter_dimensionless_input(self, simple_writer, input_data):
        """Check that fields with expected dimensionless input are accepted."""
        if hasattr(input_data, "cosmo_factor"):
            assert input_data.cosmo_factor.scale_factor == simple_writer.scale_factor
        testfile = "setter_dimensionless_input.hdf5"
        assert simple_writer.gas._particle_ids is None
        try:
            simple_writer.gas.particle_ids = input_data
            assert isinstance(simple_writer.gas.particle_ids, cosmo_array)
            assert (
                simple_writer.gas.particle_ids.cosmo_factor.scale_factor
                == simple_writer.scale_factor
            )
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)


def test_explicit_particle_ids():
    """..."""
    raise NotImplementedError


def test_generated_particle_ids():
    """..."""
    raise NotImplementedError


def test_explicit_smoothing_lengths():
    """..."""
    raise NotImplementedError


def test_generated_smoothing_lengths():
    """..."""
    raise NotImplementedError
