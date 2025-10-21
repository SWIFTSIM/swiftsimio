from pathlib import Path
import h5py


class HandleProvider:

    def __init__(self, filename: Path, handle: h5py.File | None = None):
        self.filename = filename
        self._handle = handle

        self.handle_manager = not self._handle

    @property
    def handle(self):
        if not self._handle:
            if not self.handle_manager:
                raise RuntimeError(
                    f"File handle is managed externally but is {self._handle}."
                )
            self._handle = h5py.File(self.filename, "r")
        return self._handle

    def _close_handle_if_manager(self):
        if self.handle_manager:
            self._handle.close()
