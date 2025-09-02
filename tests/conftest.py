import os
import subprocess
import pytest

webstorage_location = (
    "https://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/ssio_ci_04_2025/"
)
test_data_location = "test_data/"


def _requires(filename):

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
def cosmological_volume(request):
    if request.param == "EagleDistributed.hdf5":
        _requires("eagle_0025.0.hdf5")
        _requires("eagle_0025.1.hdf5")
    yield _requires(request.param)


@pytest.fixture
def cosmological_volume_only_single():
    yield _requires("EagleSingle.hdf5")


@pytest.fixture
def cosmological_volume_only_distributed():
    yield _requires("EagleDistributed.hdf5")


@pytest.fixture
def cosmological_volume_dithered():
    yield _requires("LegacyCosmologicalVolumeDithered.hdf5")


@pytest.fixture
def soap_example():
    yield _requires("SoapExample.hdf5")

# Fixture for tests of remote data access with hdfstream:
# Some tests can be repeated using remote versions of the same snapshots.
server = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream"
server_test_data_path = "SWIFT/test_data/IOExamples/ssio_ci_04_2025"
@pytest.fixture(
    params=[
        # Local files
        {"filename" : f"EagleDistributed.hdf5"},
        {"filename" : f"EagleSingle.hdf5"},
        {"filename" : f"LegacyCosmologicalVolume.hdf5"},
        # Remote files
        {"filename" : f"{server_test_data_path}/EagleDistributed.hdf5", "server" : server},
        {"filename" : f"{server_test_data_path}/EagleSingle.hdf5", "server" : server},
        {"filename" : f"{server_test_data_path}/LegacyCosmologicalVolume.hdf5", "server" : server},
    ]
)
def cosmological_volume_inc_hdfstream(request):
    """
    Return a dict of params which can be passed to swiftsimio.load()
    """
    if "server" in request.param:
        yield request.param
    else:
        yield {"filename" : _requires(request.param["filename"])}
