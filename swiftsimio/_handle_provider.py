"""Provide a mixin class for managing file handles."""

from pathlib import Path
import h5py


class HandleProvider:
    """
    Mixin class for managing file handles.

    Provides a pattern that we use across other classes to either open a file handle or
    receive an existing file handle. The class is aware of when it has opened the file
    handle ("owns" it), and provides a close method that will only close the handle if the
    class if it has ownership.

    Parameters
    ----------
    filename : Path
        The filename used if a handle needs to be opened.

    handle : h5py.File
        The file handle if it was opened externally.
    """

    def __init__(self, filename: Path, handle: h5py.File | None = None) -> None:
        self.filename = Path(filename)
        self._handle = handle

        self.handle_manager = not self._handle

        return

    @property
    def handle(self) -> h5py.File:
        """
        Provide a property that returns the handle, opening it if necessary.

        Returns
        -------
        h5py.File
            The file handle.

        Raises
        ------
        RuntimeError
            If the handle is not managed by this object but is falsy (e.g. it is a closed
            handle).
        """
        if not self._handle:
            if not self.handle_manager:
                raise RuntimeError(
                    f"File handle is managed externally but is {self._handle}."
                )
            self._handle = h5py.File(self.filename, "r")
        return self._handle

    def _close_handle_if_manager(self) -> None:
        """Close the file handle if this object is the manager of the handle."""
        if self.handle_manager:
            self._handle.close()

    def open_file(self):
        """"Return the file handle, re-opening if necessary"""
        return HandleProviderContextManager(self)


class HandleProviderContextManager:
    """
    Class which can be used to open the file associated with a
    HandleProvider even if it has been closed. Uses the existing
    handle if possible, or opens a new handle otherwise. Leaves the
    state of the HandleProvider unchanged.

    Implemented as a separate class in case any HandleProvider
    subclass is also a context manager.
    """
    def __init__(self, provider: HandleProvider) -> None:
        self.provider = provider
        self.handle = None

    def __enter__(self):
        if self.provider._handle:
            return self.provider._handle
        else:
            self.handle = h5py.File(self.provider.filename, "r")
            return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.close()
        return False
