"""
Define functions for handling file and dataset objects.

These are used to handle a few situations where we need to check exactly
what type of file or dataset object we have.
"""

from ._handle_provider import HandleProvider

import h5py
from pathlib import Path
from importlib.util import find_spec


def is_soft_link(obj: h5py.Group | h5py.Dataset | h5py.SoftLink) -> bool:
    """
    Return ``True`` if ``obj`` is a soft link.

    Note that soft links are usually dereferenced automatically, so to check
    if an object is a soft link a reference to the object must be obtained
    with::

      obj = group.get(key, getlink=True)

    where ``group`` is the group containing the object and ``key`` is the name
    of the object.

    Parameters
    ----------
    obj : Group, Dataset or SoftLink
        The object to check.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a soft link.
    """
    if find_spec("hdfstream") is not None:
        import hdfstream

        return isinstance(obj, (h5py.SoftLink, hdfstream.SoftLink))
    else:
        return isinstance(obj, h5py.SoftLink)


def is_dataset(obj: h5py.Group | h5py.Dataset | h5py.SoftLink) -> bool:
    """
    Return ``True`` if ``obj`` is a dataset.

    Parameters
    ----------
    obj : Group, Dataset or SoftLink
        The object to check.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a dataset.
    """
    if find_spec("hdfstream") is not None:
        import hdfstream

        return isinstance(obj, (h5py.Dataset, hdfstream.RemoteDataset))
    else:
        return isinstance(obj, h5py.Dataset)


def split_path_or_handle(obj: str | Path | h5py.File) -> tuple[Path, h5py.File]:
    """
    Given a filename or handle, return a ``(filename, handle)`` tuple.

    Parameters
    ----------
    obj : str, Path or h5py.File
        A path to a file or a file handle object.

    Returns
    -------
    tuple[Path, h5py.File]
        Tuple with the path and file handle.
    """
    if isinstance(obj, (str, Path)):
        filename = Path(obj)
        handle = None
    else:
        filename = Path(obj.filename)
        handle = obj
    return filename, handle


def open_path_or_handle(obj: str | Path | h5py.File) -> h5py.File:
    """
    Context manager to open a file, given a path or handle.

    Parameters
    ----------
    obj : str, Path or h5py.File
        A path to a file or a file handle object.

    Returns
    -------
    h5py.File
        The file handle.
    """
    filename, handle = split_path_or_handle(obj)
    return HandleProvider(filename, handle).open_file()
