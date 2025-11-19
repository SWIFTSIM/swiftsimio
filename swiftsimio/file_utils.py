"""
Define functions for handling file and dataset objects.

These are used to handle a few situations where we need to check exactly
what type of file or dataset object we have.
"""

from ._handle_provider import HandleProvider

import h5py
import hdfstream
from pathlib import Path


def is_soft_link(obj: h5py.Group | h5py.Dataset | h5py.SoftLink) -> bool:
    """
    Return True if obj is a soft link.

    Parameters
    ----------
    obj : Group, Dataset or SoftLink
        object returned by Group.get(key, getlink=True)

    Returns
    -------
    bool
        True if obj is a soft link
    """
    if hdfstream is not None and isinstance(obj, hdfstream.SoftLink):
        return True
    return isinstance(obj, h5py.SoftLink)


def is_dataset(obj: h5py.Group | h5py.Dataset | h5py.SoftLink) -> bool:
    """
    Return True if obj is a dataset.

    Parameters
    ----------
    obj : Group, Dataset or SoftLink
        object returned by Group.get(key, getlink=True)

    Returns
    -------
    bool
        True if obj is a dataset
    """
    return isinstance(obj, (hdfstream.RemoteDataset, h5py.Dataset))


def is_hdfstream_dataset(obj: h5py.Group | h5py.Dataset | h5py.SoftLink) -> bool:
    """
    Return True if obj is a hdfstream.RemoteDataset.

    Parameters
    ----------
    obj : h5py.Dataset or hdfstream.RemoteDataset
        a dataset like object in a local or remote file

    Returns
    -------
    bool
        True if obj is a hdfstream.RemoteDataset
    """
    return isinstance(obj, hdfstream.RemoteDataset)


def split_path_or_handle(obj: str | Path | h5py.File) -> tuple[Path, h5py.File]:
    """
    Given a filename or handle, return a (filename, handle) tuple.

    Parameters
    ----------
    obj : str, Path or h5py.File
        a path to a file or a file handle object

    Returns
    -------
    tuple[Path, h5py.File]
        tuple with the path and file handle
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
        a path to a file or a file handle object

    Returns
    -------
    h5py.File
        the file handle
    """
    filename, handle = split_path_or_handle(obj)
    return HandleProvider(filename, handle).open_file()
