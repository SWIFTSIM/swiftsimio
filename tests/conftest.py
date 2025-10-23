"""Test fixtures."""

import os
import subprocess
import pytest
from collections.abc import Generator
import numpy as np
import unyt
from swiftsimio import Writer
from swiftsimio.units import cosmo_units


webstorage_location = (
    "https://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/ssio_ci_04_2025/"
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

    # Box is 100 Mpc
    boxsize = 100 * unyt.Mpc

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer(cosmo_units, boxsize)

    # 32^3 particles.
    n_p = 32**3

    # Randomly spaced coordinates from 0, 100 Mpc in each direction
    x.gas.coordinates = np.random.rand(n_p, 3) * (100 * unyt.Mpc)

    # Random velocities from 0 to 1 km/s
    x.gas.velocities = np.random.rand(n_p, 3) * (unyt.km / unyt.s)

    # Generate uniform masses as 10^6 solar masses for each particle
    x.gas.masses = np.ones(n_p, dtype=float) * (1e6 * unyt.msun)

    # Generate internal energy corresponding to 10^4 K
    x.gas.internal_energy = (
        np.ones(n_p, dtype=float) * (1e4 * unyt.kb * unyt.K) / (1e6 * unyt.msun)
    )

    # Generate initial guess for smoothing lengths based on MIPS
    x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

    # If IDs are not present, this automatically generates
    x.write(test_filename)

    # Yield the test data
    yield x, test_filename

    # The file is automatically cleaned up after the test.
    os.remove(test_filename)
