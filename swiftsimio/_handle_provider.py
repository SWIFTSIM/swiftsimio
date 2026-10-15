"""Provide a mixin class for managing file handles."""

from contextlib import contextmanager
from typing import ContextManager
from pathlib import Path
import ctypes
import os
import re

import h5py
from h5py._hl.files import make_fapl

#: ``H5F_RFIC_UNUSUAL_NUM_UNUSED_NUMERIC_BITS``: accept numeric types whose
#: precision is far below their size, which SWIFT's ``DMantissa9`` and
#: ``DMantissa13`` (21 and 25 bits of an 8-byte double) are.
_RELAX_UNUSED_NUMERIC_BITS = 0x1


def _find_relax_function() -> object:
    """
    Find ``H5Pset_relax_file_integrity_checks`` in the HDF5 library h5py uses.

    h5py does not wrap this function (HDF5 >= 1.14.4), so it is called through
    ctypes in the libhdf5 already loaded into this process.

    Returns
    -------
    object
        The ctypes function, or ``None`` if it is not available.
    """
    if h5py.version.hdf5_version_tuple < (1, 14, 4):
        return None
    try:
        with open("/proc/self/maps") as maps:
            paths = {
                line.split()[-1]
                for line in maps
                if re.search(r"/libhdf5(-[^/]*)?\.so", line) and "hdf5_hl" not in line
            }
    except OSError:  # not Linux
        return None
    for path in paths:
        try:
            # RTLD_NOLOAD: only the copy h5py has loaded, never a second one
            library = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOLOAD", 4))
            function = library.H5Pset_relax_file_integrity_checks
        except (OSError, AttributeError):
            continue
        function.argtypes = [ctypes.c_int64, ctypes.c_uint64]
        function.restype = ctypes.c_int
        return function
    return None


_relax_integrity_checks = _find_relax_function()


def open_hdf5(filename: str | Path) -> h5py.File:
    """
    Open an HDF5 file for reading, accepting SWIFT's most compressed types.

    HDF5 >= 1.14.4 refuses to open datasets whose type uses far fewer bits
    than its size, such as SWIFT's ``DMantissa9`` fields, unless its
    integrity checks are relaxed for them. They are relaxed here where the
    library allows it (source files of virtual datasets inherit this);
    otherwise this is ``h5py.File(filename, "r")``.

    Parameters
    ----------
    filename : str or Path
        The file to open.

    Returns
    -------
    h5py.File
        The open file.
    """
    if _relax_integrity_checks is not None:
        try:
            fapl = make_fapl(None)
            with h5py._objects.phil:
                relaxed = _relax_integrity_checks(fapl.id, _RELAX_UNUSED_NUMERIC_BITS)
            if relaxed >= 0:
                return h5py.File(
                    h5py.h5f.open(os.fsencode(filename), h5py.h5f.ACC_RDONLY, fapl)
                )
        except Exception:
            pass  # older libraries and unsupported property lists use normal h5py
    return h5py.File(filename, "r")


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
            self._handle = open_hdf5(self.filename)
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
            with open_hdf5(self.filename) as handle:
                yield handle
