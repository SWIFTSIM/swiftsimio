"""Test fixtures."""

import os
import subprocess
import pytest
from collections.abc import Generator
import numpy as np
import unyt
from swiftsimio import Writer, cosmo_array
from swiftsimio.units import cosmo_units
import swiftsimio.metadata.particle as particle_metadata
import swiftsimio.metadata.writer.required_fields as writer_required_fields


webstorage_location = (
    "https://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/ssio_ci_11_2025/"
)
test_data_location = "test_data/"


def _requires(filename: str) -> str:
    """
    Make sure a file is present by downloading it if it's missing.

    Parameters
    ----------
    filename : str
        The name of the desired file.

    Returns
    -------
    str
        The location of the desired file.
    """
    if filename == "EagleDistributed.hdf5":
        _requires("eagle_0025.0.hdf5")
        _requires("eagle_0025.1.hdf5")

    # First check if the test data directory exists
    if not os.path.exists(test_data_location):
        os.mkdir(test_data_location)

    file_location = os.path.join(test_data_location, filename)

    if os.path.exists(file_location):
        ret = 0
    else:
        # Download it!
        ret = subprocess.call(
            ["wget", f"{webstorage_location}{filename}", "-O", file_location]
        )

    if ret != 0:
        pytest.skip(f"Unable to download file at {filename}")
        # It wrote an empty file, kill it.
        subprocess.call(["rm", file_location])

    else:
        return file_location


@pytest.fixture(
    params=[
        "EagleDistributed.hdf5",
        "EagleSingle.hdf5",
        "LegacyCosmologicalVolume.hdf5",
        "ColibreSingle.hdf5",
    ]
)
def cosmological_volume(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture provides single, distributed and legacy datasets to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires(request.param)


@pytest.fixture
def cosmological_volume_only_single() -> Generator[str, None, None]:
    """
    Fixture provides only a single (non-distributed) dataset to test on.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("EagleSingle.hdf5")


@pytest.fixture
def cosmological_volume_only_distributed() -> Generator[str, None, None]:
    """
    Fixture provides only a distributed (not monolithic) dataset to test on.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("EagleDistributed.hdf5")


@pytest.fixture
def cosmological_volume_dithered() -> Generator[str, None, None]:
    """
    Fixture provides a dithered dataset to test on.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("LegacyCosmologicalVolumeDithered.hdf5")


@pytest.fixture
def soap_example() -> Generator[str, None, None]:
    """
    Fixture provides a sample SOAP file to test on.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("SoapExample.hdf5")


@pytest.fixture(
    params=[
        "EagleDistributed.hdf5",
        "EagleSingle.hdf5",
        "LegacyCosmologicalVolume.hdf5",
        "SoapExample.hdf5",
    ]
)
def snapshot_or_soap(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture provides distributed, single & legacy, and SOAP datasets to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires(request.param)


@pytest.fixture(scope="function")
def simple_snapshot_data() -> Generator[tuple[Writer, str], None, None]:
    """
    Fixture provides a simple IC-like snapshot for testing.

    Yields
    ------
    out : Generator[tuple[Writer, str], None, None]
        The Writer object and the name of the file it wrote.
    """
    test_filename = "test_write_output_units.hdf5"

    a = 1
    # Box is 100 Mpc
    boxsize = cosmo_array(
        [100, 100, 100],
        unyt.Mpc,
        comoving=True,
        scale_factor=a,
        scale_exponent=1,
    )

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer(cosmo_units, boxsize, scale_factor=a)

    # 32^3 particles.
    n_p = 32**3

    # Randomly spaced coordinates from 0, 100 Mpc in each direction
    x.gas.coordinates = cosmo_array(
        np.random.rand(n_p, 3) * 100,
        unyt.Mpc,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )

    # Random velocities from 0 to 1 km/s
    x.gas.velocities = cosmo_array(
        np.random.rand(n_p, 3),
        unyt.km / unyt.s,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )

    # Generate uniform masses as 10^6 solar masses for each particle
    x.gas.masses = cosmo_array(
        np.ones(n_p, dtype=float) * 1e6,
        unyt.msun,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    # Generate internal energy corresponding to 10^4 K
    x.gas.internal_energy = cosmo_array(
        np.ones(n_p, dtype=float) * 1e4 / 1e6,
        unyt.kb * unyt.K / unyt.solMass,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=-2,
    )

    # Generate initial guess for smoothing lengths based on MIPS
    x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

    # If IDs are not present, this automatically generates
    x.write(test_filename)

    # Yield the test data
    yield x, test_filename

    # The file is automatically cleaned up after the test.
    os.remove(test_filename)


def _setup_extra_part_type():
    """
    Set up metadata for an extra particle type.

    Ideally this tinkering with global variables should be refactored out.
    """
    particle_metadata.particle_name_underscores["PartType7"] = "extratype"
    particle_metadata.particle_name_class["PartType7"] = "Extratype"
    particle_metadata.particle_name_text["PartType7"] = "Extratype"
    writer_required_fields.extratype = {
        "smoothing_length": {
            "handle": "SmoothingLength",
            "dimensions": unyt.dimensions.length,
        },
        **writer_required_fields._shared,
    }


def _teardown_extra_part_type():
    """
    Tear down metadata for an extra particle type.

    Ideally this tinkering with global variables should be refactored out.
    """
    particle_metadata.particle_name_underscores.pop("PartType7")
    particle_metadata.particle_name_class.pop("PartType7")
    particle_metadata.particle_name_text.pop("PartType7")
    del writer_required_fields.extratype


@pytest.fixture(scope="function")
def extra_part_type():
    """Set up and tear down metadata for an extra particle type."""
    _setup_extra_part_type()
    yield
    _teardown_extra_part_type()


@pytest.fixture(scope="function")
def write_extra_part_type():
    """
    Write a dataset with extra particle type so that we can test reading it in.

    Make sure we reliably clean up global variables and written file.
    """
    _setup_extra_part_type()
    unit_system = unyt.UnitSystem(
        name="default", length_unit=unyt.cm, mass_unit=unyt.g, time_unit=unyt.s
    )
    a = 0.5
    boxsize = cosmo_array(
        [10, 10, 10], unyt.cm, comoving=False, scale_factor=a, scale_exponent=1
    )

    x = Writer(
        unit_system,
        boxsize,
        scale_factor=a,
    )

    x.extratype.coordinates = cosmo_array(
        np.array([np.arange(10), np.zeros(10), np.zeros(10)]).astype(float).T,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    x.extratype.velocities = cosmo_array(
        np.zeros((10, 3), dtype=float),
        unyt.cm / unyt.s,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.masses = cosmo_array(
        np.ones(10, dtype=float),
        unyt.g,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.smoothing_length = cosmo_array(
        np.ones(10, dtype=float) * 5.0,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    x.write("extra_test.hdf5")
    yield
    os.remove("extra_test.hdf5")
    _teardown_extra_part_type()
