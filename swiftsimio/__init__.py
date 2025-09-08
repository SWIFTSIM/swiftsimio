from typing import Optional as _Optional, Union as _Union

from .reader import *
from .snapshot_writer import SWIFTSnapshotWriter
from .masks import SWIFTMask
from .statistics import SWIFTStatisticsFile
from .__version__ import __version__
from .__cite__ import __cite__
from .file_utils import FileOpener

import swiftsimio.metadata as metadata
import swiftsimio.accelerated as accelerated
import swiftsimio.objects as objects
from swiftsimio.objects import cosmo_array, cosmo_quantity
import swiftsimio.visualisation as visualisation
import swiftsimio.units as units
import swiftsimio.subset_writer as subset_writer
import swiftsimio.statistics as statistics

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
    filename: str,
    spatial_only: bool = True,
    safe_padding: _Union[bool, float] = True,
    server: _Optional[str] = None,
    user: _Optional[str] = None,
    password: _Optional[str] = None,
) -> SWIFTMask:
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

    server : str, optional
        server URL if opening a remote snapshot
    user : str, optional
        username if opening a remote snapshot
    password : str, optional
        password if opening a remote snapshot

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

    units = SWIFTUnits(filename, FileOpener(server, user, password))
    metadata = metadata_discriminator(filename, units)

    return SWIFTMask(
        metadata=metadata, spatial_only=spatial_only, safe_padding=safe_padding
    )


def load(
        filename: str, mask: _Optional[SWIFTMask] = None, server: _Optional[str] = None,
        user: _Optional[str] = None, password: _Optional[str] = None,
) -> SWIFTDataset:
    """
    Loads the SWIFT dataset at filename.

    Parameters
    ----------
    filename : str
        SWIFT snapshot file to read
    mask : SWIFTMask, optional
        mask to apply when reading dataset
    server : str, optional
        if not None, read files from hdfstream server
    user : str, optional
        username if opening a remote snapshot
    password : str, optional
        password if opening a remote snapshot
    """

    return SWIFTDataset(filename, FileOpener(server, user, password), mask=mask)


def load_statistics(filename: str) -> SWIFTStatisticsFile:
    """
    Loads a SWIFT statistics file (``SFR.txt``, ``energy.txt``).

    Parameters
    ----------

    filename : str
        SWIFT statistics file path
    """

    return SWIFTStatisticsFile(filename=filename)


Writer = SWIFTSnapshotWriter
