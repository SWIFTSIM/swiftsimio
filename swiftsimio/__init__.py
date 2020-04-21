from .reader import *
from .writer import SWIFTWriterDataset
from .masks import SWIFTMask
from .__version__ import __version__

import swiftsimio.metadata as metadata
import swiftsimio.accelerated as accelerated
import swiftsimio.objects as objects
import swiftsimio.visualisation as visualisation
import swiftsimio.units as units

name = "swiftsimio"


def validate_file(filename):
    """
    Checks that the provided file is a SWIFT dataset.

    Parameters
    ----------
    filename : str
        name of file we want to check is a dataset

    Return
    ------
    bool
        if `filename` is a SWIFT dataset return True,
        otherwise raise exception

    Raises
    ------
    KeyError
        Crash if the file is not a SWIFT data file
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

    Parameters
    ----------
    filename : str
        SWIFT data file to read from
    spatial_only : bool, optional
        Flag for only spatial masking, this is much faster but will not 
        allow you to use masking on other variables (e.g. density). 
        Defaults to True.

    Returns
    -------
    SWIFTMask
        empty mask object set up with the correct units and metadata

    Notes
    -----
    If you are only planning on using this as a spatial mask, ensure
    that spatial_only remains true. If you require the use of the
    constrain_mask function, then you will need to use the (considerably
    more expensive, ~bytes per particle instead of ~bytes per cell
    spatial_only=False version).
    """

    units = SWIFTUnits(filename)
    metadata = SWIFTMetadata(filename, units)

    return SWIFTMask(metadata=metadata, spatial_only=spatial_only)


def load(filename, mask=None) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.

    Parameters
    ----------
    filename : str
        file to containing SWIFT dataset to read
    mask : SWIFTMask, optional
        mask to apply when reading dataset
    """

    return SWIFTDataset(filename, mask=mask)


# Rename this object to something simpler.
Writer = SWIFTWriterDataset
