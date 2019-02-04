from .reader import *
from .writer import SWIFTWriterDataset

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


def load(filename) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename)


# Rename this object to something simpler.
Writer = SWIFTWriterDataset
