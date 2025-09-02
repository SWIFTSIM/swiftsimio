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

#
# Allow enabling remote file tests with a command line flag
#
def pytest_addoption(parser):
    parser.addoption(
        "--enable-hdfstream-tests", action="store_true", default=False, help="Run tests which access files using the hdfstream module"
    )

#
# Fixtures for tests of remote data access with hdfstream:
# Some tests can be repeated using remote versions of the same snapshots.
#
server = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream"
server_test_data_path = "Tests/SWIFT/IOExamples/ssio_ci_04_2025"
def test_data_parameters(request):
    if "server" in request.param:
        if not request.config.getoption("--enable-hdfstream-tests"):
            pytest.skip("Skipping remote tests: --enable-hdfstream-tests not set")
        return request.param
    else:
        return {"filename" : _requires(request.param["filename"])}

# Fixture which returns load parameters for the cosmological volume
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
def cosmological_volume_params(request):
    return test_data_parameters(request)

@pytest.fixture(
    params=[
        {"filename" : f"LegacyCosmologicalVolumeDithered.hdf5"},
        {"filename" : f"{server_test_data_path}/LegacyCosmologicalVolumeDithered.hdf5", "server" : server},
    ]
)
def cosmological_volume_dithered_params(request):
    return test_data_parameters(request)

@pytest.fixture(
    params=[
        {"filename" : f"SoapExample.hdf5"},
        {"filename" : f"{server_test_data_path}/SoapExample.hdf5", "server" : server},
    ]
)
def soap_example_params(request):
    return test_data_parameters(request)
