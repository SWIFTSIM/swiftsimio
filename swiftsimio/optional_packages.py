"""
Import optional packages.

This includes:

+ tqdm: progress bars
+ scipy.spatial: KDTrees
+ numba/cuda: visualisation
"""

from typing import Iterable, Callable, Any

# TQDM
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):

    def tqdm(x: Iterable, *args: tuple[Any], **kwargs: dict[str, Any]) -> Iterable:
        """
        Mock the main tqdm function for use if it's unavailable.

        Parameters
        ----------
        x : Iterable
            The iterable whose progress would be track by tqdm.

        *args : tuple[Any]
            Arbitrary additional arguments.

        **kwargs : dict[str, Any]
            Arbitrary additional kwargs.

        Returns
        -------
        out : Iterable
            The input iterable is returned.
        """
        return x

    TQDM_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as KDTree

    TREE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KDTree = None
    TREE_AVAILABLE = False


# Astropy
try:
    import astropy

    ASTROPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    astropy = None
    ASTROPY_AVAILABLE = False


# matplotlib
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    plt = None
    MATPLOTLIB_AVAILABLE = False

# numba
try:
    from numba import jit, prange
    from numba.core.config import NUMBA_NUM_THREADS as NUM_THREADS
except (ImportError, ModuleNotFoundError):
    try:
        from numba import jit, prange
        from numba.config import NUMBA_NUM_THREADS as NUM_THREADS
    except (ImportError, ModuleNotFoundError):
        print(
            "You do not have numba installed. Please consider installing "
            "if you are going to be doing visualisation or indexing large arrays "
            "(pip install numba)"
        )

        def jit(*args: tuple[Any], **kwargs: dict[str, Any]) -> Callable:
            """
            Mock the numba jit function for use if not available.

            Parameters
            ----------
            *args : tuple[Any]
                Arbitrary arguments.

            **kwargs : dict[str, Any]
                Arbitrary kwargs.

            Returns
            -------
            out : Callable
                The wrapper function (a trivial wrapper).
            """
            return lambda func: func

        prange = range
        NUM_THREADS = 1

# Numba/CUDA
try:
    from numba.cuda.cudadrv.error import CudaSupportError

    try:
        import numba.cuda.cudadrv.driver as drv
        from numba import cuda
        from numba.cuda import jit as cuda_jit

        try:
            CUDA_AVAILABLE = cuda.is_available()
        except AttributeError:
            # Backwards compatibility with older versions
            # Check for the driver

            d = drv.Driver()
            d.initialize()

            CUDA_AVAILABLE = True

    except CudaSupportError:
        CUDA_AVAILABLE = False

except (ImportError, ModuleNotFoundError):
    # Mock the CudaSupportError so that we can raise it in cases
    # where we don't have numba installed.

    class CudaSupportError(Exception):
        """
        Mock the CudaSupportError class for use if it's unavailable.

        Parameters
        ----------
        message : str
            The error message.
        """

        def __init__(self, message: str) -> None:
            self.message = message

    CUDA_AVAILABLE = False


if not CUDA_AVAILABLE:
    # Mock cuda-jit to prevent crashes
    def cuda_jit(*args, **kwargs):  # NOQA
        """
        Mock the cuda_jit function for use if it's unavailable.

        Returns
        -------
        out : Callable
            The wrapper function (a trivial wrapper).
        """
        return lambda func: func

    # For additional CUDA API access
    cuda = None
