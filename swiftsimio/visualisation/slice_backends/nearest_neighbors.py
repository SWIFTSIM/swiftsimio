from numpy import (
    float64,
    float32,
    ndarray,
)

from swiftsimio.accelerated import jit, prange, NUM_THREADS

@jit(nopython=True, fastmath=True)
def slice_scatter(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
    box_x: float64 = 0.0,
    box_y: float64 = 0.0,
    box_z: float64 = 0.0,
) -> ndarray:
    pass


@jit(nopython=True, fastmath=True, parallel=True)
def slice_scatter_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
    box_x: float64 = 0.0,
    box_y: float64 = 0.0,
    box_z: float64 = 0.0,
) -> ndarray:
    pass
