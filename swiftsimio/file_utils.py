from ._handle_provider import HandleProvider

import h5py
import hdfstream
from pathlib import Path


def is_soft_link(obj):
    """
    Return True if obj is a soft link

    Parameters
    ----------

    obj : Group, Dataset or SoftLink
        object returned by Group.get(key, getlink=True)
    """
    if hdfstream is not None and isinstance(obj, hdfstream.SoftLink):
        return True
    return isinstance(obj, h5py.SoftLink)


def is_dataset(obj):
    """
    Return True if obj is a dataset

    Parameters
    ----------

    obj : Group, Dataset or SoftLink
        object returned by Group.get(key, getlink=True)
    """
    if hdfstream is not None and isinstance(obj, hdfstream.RemoteDataset):
        return True
    return isinstance(obj, h5py.Dataset)


def is_hdfstream_dataset(obj):
    """
    Return True if obj is a hdfstream.RemoteDataset

    Parameters
    ----------

    obj : h5py.Dataset or hdfstream.RemoteDataset
        a dataset like object in a local or remote file
    """
    if hdfstream is None:
        return False
    else:
        return isinstance(obj, hdfstream.RemoteDataset)


def open_path_or_handle(obj):
    """
    Context manager to open a file, given a path or handle
    """
    if isinstance(obj, (str, Path)):
        filename = Path(obj)
        handle = None
    else:
        filename = Path(obj.filename)
        handle = obj
    return HandleProvider(filename, handle).open_file()
