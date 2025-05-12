import os
import subprocess
import pytest

webstorage_location = "https://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/"
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


@pytest.fixture
def cosmo_volume_example():
    yield _requires("cosmo_volume_example.hdf5")


@pytest.fixture
def cosmological_volume():
    yield _requires("cosmological_volume.hdf5")


@pytest.fixture
def cosmological_volume_dithered():
    yield _requires("cosmological_volume_dithered.hdf5")


@pytest.fixture
def soap_example():
    yield _requires("soap_example.hdf5")
