import h5py

try:
    import hdfstream
except ImportError:
    hdfstream = None


class FileOpener:
    def __init__(self, server=None):
        self.server = server
        if server is not None:
            self.root = hdfstream.open(
                server, "/", max_depth=3, data_size_limit=10 * 1024 * 1024
            )

    def open(self, filename, mode="r"):
        if self.server is None:
            return h5py.File(filename, mode)
        else:
            return self.root[filename]


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
        return isinstance(handle, hdfstream.RemoteDataset)

