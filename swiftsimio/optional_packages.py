"""
Imports of optional packages.

This includes:

+ tqdm: progress bars
+ scipy.spatial: KDTrees
+ sphviewer: visualisation
+ numba/cuda: visualisation
"""

# TQDM
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):

    def tqdm(x, *args, **kwargs):
        return x

    TQDM_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as KDTree

    TREE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KDTree = None
    TREE_AVAILABLE = False


# Py-sphviewer
try:
    import sphviewer as viewer

    SPHVIEWER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    viewer = None
    SPHVIEWER_AVAILABLE = False


# Astropy
try:
    import astropy

    ASTROPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    astropy = None
    ASTROPY_AVAILABLE = False


# Numba/CUDA
try:
    from numba.cuda.cudadrv.error import CudaSupportError

    try:
        import numba.cuda.cudadrv.driver as drv
        from numba import cuda
        from numba.cuda import jit as cuda_jit

        d = drv.Driver()
        d.initialize()

        CUDA_AVAILABLE = True
    # Check for the driver
    except CudaSupportError:
        CUDA_AVAILABLE = False

except (ImportError, ModuleNotFoundError):
    # Mock the CudaSupportError so that we can raise it in cases
    # where we don't have numba installed.

    class CudaSupportError(Exception):
        def __init__(self, message):
            self.message = message

    CUDA_AVAILABLE = False


if not CUDA_AVAILABLE:
    # Mock cuda-jit to prevent crashes
    def cuda_jit(*args, **kwargs):
        def x(func):
            return func

        return x

    # For additional CUDA API access
    cuda = None
