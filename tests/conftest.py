"""Test fixtures."""

import os
import warnings
import subprocess
import itertools
import pytest
from pytest import TempPathFactory
from pathlib import Path
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import unyt
import h5py
from swiftsimio import Writer, cosmo_array
import swiftsimio.metadata.particle as particle_metadata
import swiftsimio.metadata.writer.required_fields as writer_required_fields
from .helper import create_minimal_writer, create_two_type_writer
from swiftsimio.optional_packages import HDFSTREAM_AVAILABLE, hdfstream

# URL to download the test data
webstorage_location = (
    "https://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/ssio_ci_04_2026/"
)

# Where to write the downloaded files
test_data_location = "test_data/"


def pytest_addoption(parser):
    """
    Define command line flags to set the server URL and path to test data.

    To test against the server on Cosma, use::

        --hdfstream-server=https://dataweb.cosma.dur.ac.uk:8443/hdfstream
        --hdfstream-prefix=Tests/SWIFT/IOExamples/ssio_ci_04_2025

    To test against a local server (e.g. in a github workflow), use::

        --hdfstream-server=http://localhost:8080/hdfstream
        --hdfstream-prefix=test_data

    Omit the server URL to skip remote file tests.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser.
    """
    parser.addoption(
        "--hdfstream-server", default=None, help="Hdfstream server URL for tests"
    )
    parser.addoption(
        "--hdfstream-prefix",
        default=test_data_location,
        help="Directory with test data on the server",
    )


def _download(filename: str) -> bool:
    """
    Download a test data file if it's missing.

    The file is downloaded to a temporary name and moved into place once
    complete.

    Parameters
    ----------
    filename : str
        The name of the file to download.

    Returns
    -------
    bool
        ``True`` if the file is present after the call, ``False`` otherwise.
    """
    os.makedirs(test_data_location, exist_ok=True)

    file_location = os.path.join(test_data_location, filename)

    if os.path.exists(file_location):
        return True

    partial_location = f"{file_location}.part"
    ret = subprocess.call(
        ["wget", f"{webstorage_location}{filename}", "-O", partial_location]
    )

    if ret != 0:
        # It may have written an empty or partial file, kill it.
        if os.path.exists(partial_location):
            os.remove(partial_location)
        return False

    os.replace(partial_location, file_location)
    return True


def _requires(filename: str) -> str:
    """
    Make sure a file is present, and skip the test if it's missing.

    Files are downloaded once at the start of the test session by
    :func:`preload_test_data`, missing files mean the download failed.

    Parameters
    ----------
    filename : str
        The name of the desired file.

    Returns
    -------
    str
        The location of the desired file.
    """
    if HDFSTREAM_AVAILABLE and isinstance(filename, hdfstream.RemoteFile):
        return filename

    if filename == "EagleDistributed.hdf5":
        _requires("eagle_0025.0.hdf5")
        _requires("eagle_0025.1.hdf5")

    file_location = os.path.join(test_data_location, filename)

    if not os.path.exists(file_location):
        pytest.skip(f"Test data file {filename} unavailable")

    return file_location


def preload_test_data(max_workers: int = 1):
    """
    Download all test data in advance.

    This runs at the start of the test session (see :func:`pytest_sessionstart`)
    so that test workers running in parallel don't compete to download files.

    Tests using the hdfstream service are implemented by running a local
    copy of the server in a container when run through github actions.
    In that case we need to ensure that all files are present before the
    server is started, so this is also called directly.

    Parameters
    ----------
    max_workers : int, optional
        The number of files to download concurrently.
    """
    all_filenames = [
        "EagleDistributed.hdf5",
        "eagle_0025.0.hdf5",
        "eagle_0025.1.hdf5",
        "EagleSingle.hdf5",
        "LegacyCosmologicalVolume.hdf5",
        "LegacyCosmologicalVolumeDithered.hdf5",
        "SoapExample.hdf5",
        "ColibreSingle.hdf5",
        "ColibreLOS.hdf5",
    ]
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_download, all_filenames))
    else:
        for name in all_filenames:
            _download(name)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session):
    """
    Download test data before any tests run.

    With ``pytest-xdist`` this runs only in the controller process, before the
    workers are started. Files are downloaded concurrently.

    Parameters
    ----------
    session : pytest.Session
        The pytest session.
    """
    if not hasattr(session.config, "workerinput"):
        n = session.config.getoption("numprocesses", default=None) or 1
        preload_test_data(max_workers=n)


def open_local_with_filename(filename: str, request: pytest.FixtureRequest) -> str:
    """
    Return the name of a local HDF5 file to read.

    Parameters
    ----------
    filename : str
        The name of the file.
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Returns
    -------
    str
        The name of the file.
    """
    return _requires(filename)


def open_local_with_handle(filename: str, request: pytest.FixtureRequest) -> h5py.File:
    """
    Return an open h5py.File to read.

    Parameters
    ----------
    filename : str
        The name of the file.
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Returns
    -------
    h5py.File
        The open file.
    """
    return h5py.File(_requires(filename), "r")


def open_with_hdfstream(
    filename: str, request: pytest.FixtureRequest
) -> "hdfstream.RemoteFile":
    """
    Return an open :class:`hdfstream.RemoteFile` to read.

    Skips the test if no server URL was specified or the :mod:`hdfstream`
    module cannot be imported.

    Parameters
    ----------
    filename : str
        The name of the file.
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Returns
    -------
    hdfstream.RemoteFile
        The open file.
    """
    if not HDFSTREAM_AVAILABLE:
        pytest.skip("hdfstream module could not be imported")
    server = request.config.getoption("--hdfstream-server")
    prefix = request.config.getoption("--hdfstream-prefix")
    if server is None:
        pytest.skip("hdfstream server URL not specified")
    return hdfstream.open(server, f"{prefix}/{filename}")


access_methods = [
    open_local_with_filename,
    open_local_with_handle,
    open_with_hdfstream,
]


@pytest.fixture(
    params=itertools.product(
        [
            "EagleDistributed.hdf5",
            "EagleSingle.hdf5",
            "LegacyCosmologicalVolume.hdf5",
            "ColibreSingle.hdf5",
        ],
        access_methods,
    )
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
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    filename, access_method = request.param
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield access_method(filename, request)


@pytest.fixture(params=access_methods)
def cosmological_volume_only_single(
    request: pytest.FixtureRequest,
) -> Generator[str, None, None]:
    """
    Fixture provides only a single (non-distributed) dataset to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    out : Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield request.param("EagleSingle.hdf5", request)


@pytest.fixture
def cosmological_volume_only_single_local() -> Generator[str, None, None]:
    """
    Fixture provides only a single (non-distributed) dataset to test on.

    This version only opens the file by name. Tests using this fixture will
    only run once and will not try passing in an already open h5py.File or
    a hdfstream.RemoteFile. This is intended for expensive tests where we
    are not testing the file access mechanism.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("EagleSingle.hdf5")


@pytest.fixture
def soap_only_local() -> Generator[str, None, None]:
    """
    Fixture provides only a SOAP dataset.

    This verison only opens the file by name. Tests using this fixture will
    only run once and will not try passing in an already open h5py.File or
    a hdfstream.RemoteFile. This is intended for tests where we are not
    testing the file access mechanism.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield _requires("SoapExample.hdf5")


@pytest.fixture(params=access_methods)
def cosmological_volume_only_distributed(
    request: pytest.FixtureRequest,
) -> Generator[str, None, None]:
    """
    Fixture provides only a distributed (not monolithic) dataset to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield request.param("EagleDistributed.hdf5", request)


@pytest.fixture(
    params=[
        open_local_with_filename,
    ]
)
def cosmological_volume_dithered(
    request: pytest.FixtureRequest,
) -> Generator[str, None, None]:
    """
    Fixture provides a dithered dataset to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield request.param("LegacyCosmologicalVolumeDithered.hdf5", request)


@pytest.fixture(params=access_methods)
def soap_example(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture provides a sample SOAP file to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    yield request.param("SoapExample.hdf5", request)


@pytest.fixture(params=access_methods)
def los_example(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture provides a sample line of sight file to test on.

    Parameters
    ----------
    request : FixtureRequest
        Parameter value(s) from the fixture.

    Yields
    ------
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield request.param("ColibreLOS.hdf5", request)


@pytest.fixture(
    params=itertools.product(
        [
            "EagleDistributed.hdf5",
            "EagleSingle.hdf5",
            "LegacyCosmologicalVolume.hdf5",
            "SoapExample.hdf5",
        ],
        access_methods,
    )
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
    Generator[str, None, None]
        The file name, after downloading if required.
    """
    filename, access_method = request.param
    yield access_method(filename, request)


@pytest.fixture(scope="function")
def simple_writer() -> Generator[Writer, None, None]:
    """
    Provide a :class:`~swiftsimio.snapshot_writer.Writer` with required gas fields.

    Yields
    ------
    Generator[Writer, None, None]
        The Writer object.
    """
    yield create_minimal_writer()


@pytest.fixture(scope="function")
def two_type_writer() -> Generator[Writer, None, None]:
    """
    Provide a :class:`~swiftsimio.snapshot_writer.Writer` with required gas & DM fields.

    Yields
    ------
    Generator[Writer, None, None]
        The Writer object.
    """
    yield create_two_type_writer()


@pytest.fixture(scope="function")
def simple_snapshot_data(
    tmp_path_factory: TempPathFactory,
) -> Generator[tuple[Writer, Path], None, None]:
    """
    Provide a simple IC-like snapshot for testing.

    Parameters
    ----------
    tmp_path_factory : TempPathFactory
        Pytest fixture to create temporary directories.

    Yields
    ------
    Generator[tuple[Writer, Path], None, None]
        The Writer object and the name of the file it wrote.
    """
    test_filename = (
        tmp_path_factory.mktemp("simple_snapshot") / "test_write_output_units.hdf5"
    )
    w = create_minimal_writer()

    w.write(test_filename)

    yield w, test_filename

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
        "smoothing_lengths": {
            "handle": "SmoothingLengths",
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
def write_extra_part_type(
    tmp_path_factory: TempPathFactory,
) -> Generator[Path, None, None]:
    """
    Write a dataset with extra particle type so that we can test reading it in.

    Make sure we reliably clean up global variables and written file.

    Parameters
    ----------
    tmp_path_factory : TempPathFactory
        Pytest fixture to create temporary directories.

    Yields
    ------
    Generator[Path, None, None]
        The name of the file written.
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
        unit_system=unit_system,
        boxsize=boxsize,
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

    x.extratype.smoothing_lengths = cosmo_array(
        np.ones(10, dtype=float) * 5.0,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    testfile = tmp_path_factory.mktemp("extra_part_type") / "extra_test.hdf5"
    x.write(testfile)
    yield testfile
    os.remove(testfile)
    _teardown_extra_part_type()
