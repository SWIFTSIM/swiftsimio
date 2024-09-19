"""
Tests that we can open SOAP files
"""

from tests.helper import requires

from swiftsimio import load


@requires("soap_example.hdf5")
def test_soap_can_load(filename):
    data = load(filename)

    return
