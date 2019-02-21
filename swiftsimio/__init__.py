from .reader import *
from .writer import SWIFTWriterDataset
from .masks import SWIFTMask

import swiftsimio.metadata as metadata

name = "swiftsimio"


def validate_file(filename):
    """
    Checks that the provided file is a SWIFT dataset.
    """
    try:
        with h5py.File(filename, "r") as handle:
            if handle["Code"].attrs["Code"] != b"SWIFT":
                raise KeyError
    except KeyError:
        raise IOError("File is not of SWIFT data type")

    return True


def mask(filename) -> SWIFTMask:
    """
    Sets up a masking object for you to use with the correct units and
    metadata available.
    """

    units = SWIFTUnits(filename)
    metadata = SWIFTMetadata(filename, units)

    return SWIFTMask(metadata=metadata)


def load(filename, mask=None) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename, mask=mask)


# Rename this object to something simpler.
Writer = SWIFTWriterDataset
