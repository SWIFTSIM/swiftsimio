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
