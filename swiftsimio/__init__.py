from .reader import *

import swiftsimio.metadata as metadata


def load(filename) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename)