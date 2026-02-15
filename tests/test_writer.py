"""Test the :class:`~swiftsimio.snapshot_writer.Writer`."""

import os
import pytest
import numpy as np
from swiftsimio import load, cosmo_array, Writer
from .helper import create_minimal_writer
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


def test_scale_factor_written(simple_snapshot_data):
    """Check that the scale factor gets written in the output."""
    testfile = "scale_factor_written.hdf5"
    w = create_minimal_writer(a=0.5)
    w.write(testfile)
    dat = load(testfile)
    assert w.scale_factor != 1.0  # make sure it's not just a default value
    assert dat.metadata.a == w.scale_factor


def test_write_non_required_field(simple_writer):
    """
    Try to write a non-required field with the writer.

    Expectation is that it is silently ignored and not written out. Written file should
    still be readable.

    May be desirable to change this behaviour in the future, perhaps using TypedDict.
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
        # expect not to write a file but clean up if the unexpected happens
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
                    np.ones(100),
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
            with pytest.raises(AssertionError, match="The scale factor of masses"):
                w.gas.masses = cosmo_array(
                    np.ones(100),
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
                w.gas.masses = np.ones(100) * u.solMass
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
                w.gas.masses = np.ones(100)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)

    @pytest.mark.parametrize(
        "input_data_generator",
        (
            lambda n_p: np.arange(1, n_p + 1, dtype=int),
            lambda n_p: np.arange(1, n_p + 1, dtype=int) * u.dimensionless,
            lambda n_p: cosmo_array(
                np.arange(1, n_p + 1, dtype=int),
                u.dimensionless,
                comoving=True,
                scale_factor=1.0,
                scale_exponent=0,
            ),
        ),
    )
    def test_setter_dimensionless_input(self, simple_writer, input_data_generator):
        """Check that fields with expected dimensionless input are accepted."""
        n_p = simple_writer.gas.masses.size
        input_data = input_data_generator(n_p)
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
            simple_writer.write(testfile)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)


@pytest.mark.parametrize(
    "input_data_generator",
    (
        lambda n_p: np.arange(n_p, dtype=int) + 1000,
        lambda n_p: (np.arange(n_p, dtype=int) + 1000) * u.dimensionless,
        lambda n_p: cosmo_array(
            np.arange(n_p, dtype=int) + 1000,
            u.dimensionless,
            comoving=True,
            scale_factor=1.0,
            scale_exponent=0,
        ),
    ),
)
def test_explicit_particle_ids(input_data_generator, simple_writer):
    """
    Check that particle IDs can be set directly (i.e. not auto-generated).

    The IDs used in this test are offset by 1000 to make sure we didn't use auto-generated
    IDs instead of the ones explicitly provided.
    """
    testfile = "explicit_particle_ids.hdf5"
    assert simple_writer.gas._particle_ids is None
    n_p = simple_writer.gas.masses.size
    input_data = input_data_generator(n_p)
    try:
        simple_writer.gas.particle_ids = input_data
        assert isinstance(simple_writer.gas.particle_ids, cosmo_array)
        assert (
            simple_writer.gas.particle_ids.cosmo_factor.scale_factor
            == simple_writer.scale_factor
        )
        simple_writer.write(testfile)
        dat = load(testfile)
        assert (
            dat.gas.particle_ids.to_comoving_value(u.dimensionless).astype(int)
            == input_data.view(np.ndarray)
        ).all()
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)


def test_duplicate_ids_invalid(simple_writer):
    """Check that duplicate particle IDs are rejected."""
    assert simple_writer.gas._particle_ids is None
    with pytest.raises(
        ValueError, match="gas.particle_ids must not have repeated IDs."
    ):
        simple_writer.gas.particle_ids = np.ones(simple_writer.gas.masses.size)


def test_duplicate_ids_across_types_invalid(two_type_writer):
    """Check that duplicate particle IDs in different particle types are rejected."""
    testfile = "duplicate_ids_across_types_invalid.hdf5"
    assert two_type_writer.gas._particle_ids is None
    assert two_type_writer.dark_matter._particle_ids is None
    two_type_writer.gas.particle_ids = np.arange(1, two_type_writer.gas.masses.size + 1)
    two_type_writer.dark_matter.particle_ids = np.arange(
        1, two_type_writer.dark_matter.masses.size + 1
    )
    try:
        with pytest.raises(
            ValueError, match="Particle IDs must be unique across all particle types."
        ):
            two_type_writer.write(testfile)
    finally:
        # expect not to write a file but clean up if the unexpected happens
        if os.path.exists(testfile):
            os.remove(testfile)


@pytest.mark.parametrize("invalid_id", (0, -1))
def test_invalid_ids(invalid_id, simple_writer):
    """Check that a particle ID with a 0 or negative value is rejected."""
    invalid_ids = np.arange(1, simple_writer.gas.masses.size + 1)  # valid for now
    invalid_ids[0] = invalid_id  # now invalid
    assert np.unique(invalid_ids).size == invalid_ids.size
    with pytest.raises(ValueError, match="gas.particle_ids must be >= 1."):
        simple_writer.gas.particle_ids = invalid_ids


class TestGeneratedParticleIDs:
    """Test automatically generated particle IDs."""

    def test_generated_particle_ids(self, simple_writer):
        """Check that particle ID generation produces sensible IDs."""
        testfile = "generated_particle_ids.hdf5"
        assert simple_writer.gas._particle_ids is None
        assert simple_writer.gas.check_consistent()
        assert simple_writer.gas.requires_particle_ids_before_write
        try:
            simple_writer.write(testfile)
            dat = load(testfile)
            assert dat.gas.particle_ids.size == simple_writer.gas.masses.size
            assert np.unique(dat.gas.particle_ids).size == dat.gas.particle_ids.size
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_generated_particle_ids_multiple_types(self, two_type_writer):
        """Check that all IDs are unique for multiple types."""
        testfile = "generated_particle_ids_multiple_types.hdf5"
        assert two_type_writer.gas._particle_ids is None
        assert two_type_writer.dark_matter._particle_ids is None
        try:
            two_type_writer.write(testfile)
            dat = load(testfile)
            assert (
                np.unique(
                    np.concatenate((dat.gas.particle_ids, dat.dark_matter.particle_ids))
                ).size
                == dat.gas.masses.size + dat.dark_matter.masses.size
            )
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)

    def test_overwrite_particle_ids(self, two_type_writer):
        """
        Check that we overwrite particle_ids when they are not present for all types.

        If user provides IDs for some types and not others we don't bother working out
        what IDs are available to assign to particle that don't have IDs, we just
        overwrite everything. However, if we overwrite any user-provided IDs, we should
        warn.
        """
        testfile = "overwrite_particle_ids.hdf5"
        two_type_writer.gas.particle_ids = np.arange(
            1, two_type_writer.gas.masses.size + 1
        )
        assert two_type_writer.dark_matter._particle_ids is None
        try:
            with pytest.warns(RuntimeWarning, match="Overwriting gas.particle_ids"):
                two_type_writer.write(testfile)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)


def test_explicit_smoothing_lengths(simple_writer):
    """Check that smoothing lengths can be explicitly passed in."""
    testfile = "explicit_smoothing_lengths.hdf5"
    simple_writer.gas._smoothing_lengths = None  # ensure they are blank
    simple_writer.gas.smoothing_lengths = cosmo_array(
        np.ones(simple_writer.gas.masses.size),
        u.kpc,
        comoving=True,
        scale_factor=simple_writer.scale_factor,
        scale_exponent=1,
    )
    try:
        simple_writer.write(testfile)
        dat = load(testfile)
        assert np.allclose(
            dat.gas.smoothing_lengths, simple_writer.gas.smoothing_lengths
        )
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)


def test_generated_smoothing_lengths(two_type_writer):
    """
    Check that we can automatically generate smoothing lengths.

    This should only work for types where smoothing lengths are a required field.
    """
    testfile = "generated_smoothing_lengths.hdf5"
    two_type_writer.gas._smoothing_lengths = None  # ensure they are blank
    two_type_writer.gas.generate_smoothing_lengths()
    assert two_type_writer.gas._smoothing_lengths is not None
    with pytest.raises(
        RuntimeError,
        match="Cannot generate smoothing lengths for particle types that don't require "
        "them.",
    ):
        two_type_writer.dark_matter.generate_smoothing_lengths()
    try:
        two_type_writer.write(testfile)
        dat = load(testfile)
        assert np.allclose(
            dat.gas.smoothing_lengths, two_type_writer.gas.smoothing_lengths
        )
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)
