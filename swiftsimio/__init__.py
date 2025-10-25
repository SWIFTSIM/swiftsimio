from pathlib import Path
import h5py

from .reader import *
from .snapshot_writer import SWIFTSnapshotWriter
from .masks import SWIFTMask
from .statistics import SWIFTStatisticsFile
from .__version__ import __version__
from .__cite__ import __cite__

import swiftsimio.metadata as metadata
import swiftsimio.accelerated as accelerated
import swiftsimio.objects as objects
from swiftsimio.objects import cosmo_array, cosmo_quantity
import swiftsimio.visualisation as visualisation
import swiftsimio.units as units
import swiftsimio.subset_writer as subset_writer
import swiftsimio.statistics as statistics
from swiftsimio.metadata.objects import _metadata_discriminator

name = "swiftsimio"


def validate_file(filename: str):
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


def mask(
    filename: str | Path,
    spatial_only: bool = True,
    safe_padding: bool | float = True,
) -> SWIFTMask:
    """
    Sets up a masking object for you to use with the correct units and
    metadata available.

    Parameters
    ----------
    filename : str or Path
        SWIFT data file to read from. Can also be an open h5py.File handle.

    spatial_only : bool, optional
        Flag for only spatial masking, this is much faster but will not
        allow you to use masking on other variables (e.g. density).
        Defaults to True.

    safe_padding : bool or float, optional
        If snapshot does not specify bounding box of cell particles (MinPositions &
        MaxPositions), pad the mask to gurantee that *all* particles in requested
        spatial region(s) are selected. If the bounding box metadata is present, this
        argument is ignored. The default (``True``) is to pad by one cell length.
        Padding can be disabled (``False``) or set to a different fraction of the
        cell length (e.g. ``0.2``). Only entire cells are loaded, but if the region
        boundary is more than ``safe_padding`` from a cell boundary the neighbouring
        cell is not read. Switching off can reduce I/O load by up to a factor of 10
        in some cases (but a few particles in region could be missing). See
        https://swiftsimio.readthedocs.io/en/latest/masking/index.html for further
        details.

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
    if isinstance(filename, str):
        filename = Path(filename)
    with h5py.File(filename, "r") as handle:
        units = SWIFTUnits(filename, handle=handle)
        metadata = _metadata_discriminator(filename, units, handle=handle)
        mask = SWIFTMask(
            filename,
            metadata=metadata,
            spatial_only=spatial_only,
            safe_padding=safe_padding,
            handle=handle,
        )
    return mask


def load(filename: str | Path, mask: SWIFTMask | None = None) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.

    Parameters
    ----------
    filename : str or Path
        SWIFT data file to read from. Can also be an open h5py.File handle.

    mask : SWIFTMask, optional
        mask to apply when reading dataset

    Returns
    -------
    SWIFTDataset
        dataset object providing an interface to the data file.
    """
    if isinstance(filename, str):
        filename = Path(filename)

    with h5py.File(filename, "r") as handle:
        data = SWIFTDataset(filename, mask=mask, handle=handle)

    return data


def load_statistics(filename: str | Path) -> SWIFTStatisticsFile:
    """
    Loads a SWIFT statistics file (``SFR.txt``, ``energy.txt``).

    Parameters
    ----------

    filename : str or Path
        SWIFT statistics file path
    """

    return SWIFTStatisticsFile(filename=filename)


Writer = SWIFTSnapshotWriter
