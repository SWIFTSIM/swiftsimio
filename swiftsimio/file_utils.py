import h5py
import hdfstream


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
