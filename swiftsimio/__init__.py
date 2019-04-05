from .reader import *
from .writer import SWIFTWriterDataset
from .masks import SWIFTMask

import swiftsimio.metadata as metadata
import swiftsimio.accelerated as accelerated
import swiftsimio.objects as objects

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


def mask(filename, spatial_only=True) -> SWIFTMask:
    """
    Sets up a masking object for you to use with the correct units and
    metadata available.

    If you are only planning on using this as a spatial mask, ensure
    that spatial_only remains true. If you requrie the use of the
    constrain_mask function, then you will need to use the (considerably
    more expensive, ~bytes per particle instead of ~bytes per cell
    spatial_only=False version).
    """

    units = SWIFTUnits(filename)
    metadata = SWIFTMetadata(filename, units)

    return SWIFTMask(metadata=metadata, spatial_only=spatial_only)


def load(filename, mask=None, generate_unit_func=metadata.unit_fields.generate_units, generate_cosmology_func=metadata.cosmology.cosmology_fields.generate_cosmology) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.
    """

    return SWIFTDataset(filename, mask=mask, generate_unit_func=generate_unit_func, 
            generate_cosmology_func=generate_cosmology_func)


# Rename this object to something simpler.
Writer = SWIFTWriterDataset
