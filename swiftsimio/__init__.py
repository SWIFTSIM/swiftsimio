from .reader import *
from .writer import SWIFTWriterDataset

import swiftsimio.metadata as metadata

name = "swiftsimio"


def load(filename) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename)

# Rename this object to something simpler.
Writer = SWIFTWriterDataset