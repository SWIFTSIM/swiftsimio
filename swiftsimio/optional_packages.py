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
