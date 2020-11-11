"""
Imports of optional packages.

This includes:

+ tqdm: progress bars
+ scipy.spatial: KDTrees
+ sphviewer: visualisation
"""
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


try:
    import sphviewer as viewer

    SPHVIEWER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    viewer = None
    SPHVIEWER_AVAILABLE = False


try:
    import astropy

    ASTROPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    astropy = None
    ASTROPY_AVAILABLE = False


try:
    from numba.cuda.cudadrv.error import CudaSupportError

    try:
        import numba.cuda.cudadrv.driver as drv

        d = drv.Driver()
        d.initialize()

        CUDA_AVAILABLE = True
    # Check for the driver
    except CudaSupportError:
        CUDA_AVAILABLE = False

# Check if numba is installed
except (ImportError, ModuleNotFoundError):
    CUDA_AVAILABLE = False
