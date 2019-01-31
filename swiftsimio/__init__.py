from .reader import *

import swiftsimio.metadata as metadata

name = "swiftsimio"


def load(filename) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename)