"""Provide a mixin class for managing file handles."""

from contextlib import contextmanager
from typing import ContextManager
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

    @contextmanager
    def open_file(self) -> ContextManager[h5py.File]:
        """
        Return a context manager that can be used to read the file.

        This will use the existing handle if it is open. If not, we
        assume that we're reading local HDF5 files using h5py and
        create a temporary handle.

        Returns
        -------
        ContextManager
            A context manager which can be used to read the file.
        """
        if self._handle:
            yield self._handle
        else:
            with h5py.File(self.filename, "r") as handle:
                yield handle
