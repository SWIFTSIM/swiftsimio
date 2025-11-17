import h5py
import hdfstream

from ._handle_provider import HandleProvider

class FileOpener(HandleProvider):
    """
    Context manager which will reuse an existing handle if possible, or open
    and close the file otherwise.
    """
    def __init__(self, name_or_handle, handle=None):
        if handle is None:
            if isinstance(name_or_handle, (h5py.File, hdfstream.RemoteFile)):
                filename = name_or_handle.filename
                handle = name_or_handle
            else:
                filename = name_or_handle
                handle = None
        else:
            filename = name_or_handle
        super().__init__(filename, handle=handle)

    def __enter__(self):
        return self.filename, self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_handle_if_manager()
        return False
