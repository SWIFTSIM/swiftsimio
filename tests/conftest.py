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
