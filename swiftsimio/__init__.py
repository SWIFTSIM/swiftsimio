"""
Tools for reading SWIFT simulation data.

The most used functions are :func:`~swiftsimio.load` and :func:`~swiftsimio.mask`.
The :mod:`~swiftsimio.visualisation` sub-module provides visualisation tools.
"""

from pathlib import Path

from .reader import SWIFTDataset
from .snapshot_writer import SWIFTSnapshotWriter
from .masks import SWIFTMask
from .statistics import SWIFTStatisticsFile
from .file_utils import open_path_or_handle
from .__version__ import __version__
from .__cite__ import __cite__

import swiftsimio.metadata as metadata
from swiftsimio.metadata.objects import (
    SWIFTUnits,
    SWIFTGroupMetadata,
    SWIFTSnapshotMetadata,
    SWIFTFOFMetadata,
    SWIFTSOAPMetadata,
    _metadata_discriminator,
)
import swiftsimio.accelerated as accelerated
import swiftsimio.objects as objects
from swiftsimio.objects import cosmo_array, cosmo_quantity
import swiftsimio.visualisation as visualisation
import swiftsimio.units as units
import swiftsimio.subset_writer as subset_writer
import swiftsimio.statistics as statistics

__all__ = [
    "SWIFTDataset",
    "SWIFTSnapshotWriter",
    "SWIFTMask",
    "SWIFTStatisticsFile",
    "SWIFTUnits",
    "SWIFTGroupMetadata",
    "SWIFTSnapshotMetadata",
    "SWIFTFOFMetadata",
    "SWIFTSOAPMetadata",
    "__version__",
    "__cite__",
    "metadata",
    "accelerated",
    "objects",
    "cosmo_array",
    "cosmo_quantity",
    "visualisation",
    "units",
    "subset_writer",
    "statistics",
    "name",
    "validate_file",
    "mask",
    "load",
    "load_statistics",
    "Writer",
]

name = "swiftsimio"


def validate_file(filename: str) -> bool:
    """
    Check that the provided file is a SWIFT dataset.

    Parameters
    ----------
    filename : str
        Name of file we want to check is a dataset.

    Returns
    -------
    bool
        If ``filename`` is a SWIFT dataset return ``True``,
        otherwise raise exception.

    Raises
    ------
    KeyError
        If the file is not a SWIFT data file.
    """
    try:
        with open_path_or_handle(filename) as handle:
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
    Set up a mask to apply to a :mod:`swiftsimio` dataset.

    Also makes the dataset's units and metadata available.

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
        Empty mask object set up with the correct units and metadata.

    Notes
    -----
    If you are only planning on using this as a spatial mask, ensure
    that spatial_only remains true. If you require the use of the
    constrain_mask function, then you will need to use the (considerably
    more expensive, ~bytes per particle instead of ~bytes per cell
    spatial_only=False version).
    """
    with open_path_or_handle(filename) as handle:
        units = SWIFTUnits(handle.filename, handle=handle)
        metadata = _metadata_discriminator(handle.filename, units, handle=handle)
        mask = SWIFTMask(
            handle.filename,
            metadata=metadata,
            spatial_only=spatial_only,
            safe_padding=safe_padding,
            handle=handle,
        )
    return mask


def load(filename: str | Path, mask: SWIFTMask | None = None) -> SWIFTDataset:
    """
    Load a SWIFT dataset (snapshot, FOF or SOAP catalogue).

    Parameters
    ----------
    filename : str or Path
        SWIFT data file to read from.

    mask : SWIFTMask, optional
        Mask to apply when reading dataset.

    Returns
    -------
    SWIFTDataset
        Dataset object providing an interface to the data file.
    """
    with open_path_or_handle(filename) as handle:
        data = SWIFTDataset(handle.filename, mask=mask, handle=handle)

    return data


def load_statistics(filename: str | Path) -> SWIFTStatisticsFile:
    """
    Load a SWIFT statistics file (``SFR.txt``, ``energy.txt``).

    Parameters
    ----------
    filename : str or Path
        SWIFT statistics file path.
    """
    return SWIFTStatisticsFile(filename=filename)


Writer = SWIFTSnapshotWriter
