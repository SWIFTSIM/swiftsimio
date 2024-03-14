"""
Tools for creating power spectra from SWIFT data.
"""

from numpy import float32, float64, int32, zeros, ndarray
import numpy as np
import unyt

from swiftsimio.accelerated import jit, NUM_THREADS, prange
from swiftsimio import cosmo_array
from swiftsimio.reader import __SWIFTParticleDataset

@jit(nopython=True, fastmath=True)
def deposit(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    res: int32,
    fold: float64,
    box_x: float64,
    box_y: float64,
    box_z: float64,
) -> ndarray:
    """
    Deposit the particles to a 3D grid, with the potential
    for folding. Note that unlike the other vis tools, this
    requires the 'raw' positions of the particles.

    Parameters
    ----------
    x : np.array[float64]
        array of x-positions of the particles. Must be bounded by [0, 1].
    y: np.array[float64]
        array of y-positions of the particles. Must be bounded by [0, 1].
    z: np.array[float64]
        array of z-positions of the particles. Must be bounded by [0, 1].
    m: np.array[float32]
        array of masses (or otherwise weights) of the particles
    res: int
        the number of pixels along one axis, i.e. this returns a cube
        of res * res * res.
    fold: float64
        the number of times to fold the box. Note that this is the
        number of times to fold the box, not the number of folds.
    box_x: float64
        box size in x, in the same rescaled length units as x, y, and z. c
    box_y: float64
        box size in y, in the same rescaled length units as x, y, and z.
    box_z: float64
        box size in z, in the same rescaled length units as x, y, and z.

    Returns
    -------
    np.array[float32, float32, float32]
        Pixel grid of deposited quantity density.
    """

    output = zeros((res, res, res), dtype=float32)

    float_res = float32(res)

    # Fold the box
    box_over_fold_x = box_x / fold
    box_over_fold_y = box_y / fold
    box_over_fold_z = box_z / fold

    inv_box_over_fold_x = 1.0 / box_over_fold_x
    inv_box_over_fold_y = 1.0 / box_over_fold_y
    inv_box_over_fold_z = 1.0 / box_over_fold_z

    # In pixel space.
    inv_volume_element = float_res * float_res * float_res
    inv_volume_element *= fold * fold * fold

    for x_pos, y_pos, z_pos, mass in zip(x, y, z, m):
        # Fold the particles
        x_pos = (x_pos % box_over_fold_x) * inv_box_over_fold_x
        y_pos = (y_pos % box_over_fold_y) * inv_box_over_fold_y
        z_pos = (z_pos % box_over_fold_z) * inv_box_over_fold_z

        # Convert to grid position
        x_pix = int32(x_pos * float_res)
        y_pix = int32(y_pos * float_res)
        z_pix = int32(z_pos * float_res)

        # Deposit the mass
        output[x_pix, y_pix, z_pix] += mass * inv_volume_element

    return output


@jit(nopython=True, fastmath=True, parallel=True)
def deposit_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    res: int32,
    fold: float64,
    box_x: float64,
    box_y: float64,
    box_z: float64,
) -> ndarray:
    """
    Deposit the particles to a 3D grid, with the potential
    for folding. Note that unlike the other vis tools, this
    requires the 'raw' positions of the particles.

    Parameters
    ----------
    x : np.array[float64]
        array of x-positions of the particles. Must be bounded by [0, 1].
    y: np.array[float64]
        array of y-positions of the particles. Must be bounded by [0, 1].
    z: np.array[float64]
        array of z-positions of the particles. Must be bounded by [0, 1].
    m: np.array[float32]
        array of masses (or otherwise weights) of the particles
    res: int
        the number of pixels along one axis, i.e. this returns a cube
        of res * res * res.
    fold: float64
        the number of times to fold the box. Note that this is the
        number of times to fold the box, not the number of folds.
    box_x: float64
        box size in x, in the same rescaled length units as x, y, and z. c
    box_y: float64
        box size in y, in the same rescaled length units as x, y, and z.
    box_z: float64
        box size in z, in the same rescaled length units as x, y, and z.

    Returns
    -------
    np.array[float32, float32, float32]
        Pixel grid of deposited quantity density.
    """

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = zeros((res, res, res), dtype=float32)

    for thread in prange(NUM_THREADS):
        start = thread * core_particles
        end = (thread + 1) * core_particles

        if thread == NUM_THREADS - 1:
            end = number_of_particles

        output += deposit(
            x=x[start:end],
            y=y[start:end],
            z=z[start:end],
            m=m[start:end],
            res=res,
            fold=fold,
            box_x=box_x,
            box_y=box_y,
            box_z=box_z,
        )

    return output


def render_to_deposit(
    data: __SWIFTParticleDataset,
    resolution: int,
    project: str = "masses",
    folding: float = 1.0,
    parallel: bool = False,
) -> ndarray:
    """
    Render a dataset to a deposition.

    Parameters
    ----------
    data: SWIFTDataset
        The dataset to render to a deposition.
    resolution: int
        The resolution of the deposition.
    project: str
        The quantity to project to the deposition. Must be a valid quantity
        in the dataset.
    folding: int
        The number of times to fold the box.
    parallel: bool
        Whether to use parallel deposition.

    Returns
    -------
    np.array[float32, float32, float32]
        The deposition.
    """

    # Get the positions and masses
    positions = data.coordinates
    quantity = getattr(data, project)

    if positions.comoving:
        if not quantity.compatible_with_comoving():
            raise AttributeError(
                f'Physical quantity "{project}" is not compatible with comoving coordinates!'
            )
        else:
            if not quantity.compatible_with_physical():
                raise AttributeError(
                    f'Comoving quantity "{project}" is not compatible with physical coordinates!'
                )


    # Get the box size
    box_size = data.metadata.boxsize

    if not box_size.units == positions.units:
        raise AttributeError("Box size and positions have different units!")
    
    # Deposit the particles
    arguments = dict(
        x=positions[:, 0].v,
        y=positions[:, 1].v,
        z=positions[:, 2].v,
        m=quantity.v,
        res=resolution,
        fold=folding,
        box_x=box_size[0].v,
        box_y=box_size[1].v,
        box_z=box_size[2].v,
    )
    
    if parallel:
        deposition = deposit_parallel(**arguments)
    else:
        deposition = deposit(**arguments)

    comoving = positions.comoving
    coord_cosmo_factor = positions.cosmo_factor

    units = 1.0 / (data.metadata.boxsize[0] * data.metadata.boxsize[1] * data.metadata.boxsize[2])
    units.convert_to_units(1.0 / data.metadata.boxsize.units ** 3)

    units *= quantity.units
    new_cosmo_factor = quantity.cosmo_factor / (coord_cosmo_factor ** 3)

    return cosmo_array(deposition, comoving=comoving, cosmo_factor=new_cosmo_factor, units=units)
    


def deposition_to_power_spectrum(deposition: ndarray, box_size: cosmo_array, folding: float = 1.0) -> ndarray:
    """
    Convert a deposition to a power spectrum.

    Parameters
    ----------
    deposition: np.array[float32, float32, float32]
        The deposition to convert to a power spectrum.
    box_size: cosmo_array
        The box size of the deposition, from the dataset.

    Returns
    -------
    np.array[float32]
        The power spectrum of the deposition.

    Notes
    -----

    This is adapted from Bert Vandenbroucke's code at
    https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    """

    if not len(deposition.shape) == 3:
        raise ValueError("Deposition must be a 3D array")

    if not deposition.shape[0] == deposition.shape[1] == deposition.shape[2]:
        raise ValueError("Deposition must be a cube")

    box_size = box_size[0] / folding
    npix = deposition.shape[0]
    pixel_size = box_size / npix

    # Convert to overdensity
    mean_deposition = np.mean(deposition.v)
    overdensity = (deposition.v - mean_deposition) / mean_deposition

    fft = np.fft.fftn(overdensity)
    fourier_amplitudes = (fft * fft.conj()).real

    kfreq = np.fft.fftfreq(n=npix, d=pixel_size)

    # Calculate k-value spacing (centered FFT)
    dk = 2 * np.pi / (box_size)

    # Create k-values array (adjust range based on your needs)
    kfreq = np.fft.fftfreq(npix, d=1 / dk) * npix
    kfreq3d = np.meshgrid(kfreq, kfreq, kfreq)
    knrm = np.sqrt(kfreq3d[0] ** 2 + kfreq3d[1] ** 2 + kfreq3d[2] ** 2)

    # knrm = knrm.flatten()
    # fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.linspace(kfreq.min(), kfreq.max(), npix // 2 + 1)
    #kbins = np.arange(0.5, npix//2+1, 1.)
    # kvals are in pixel co-ordinates. We know the pixels
    # span the entire box, so we can convert to physical
    # co-ordinates.
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    power_spectrum = (
        np.histogram(knrm, bins=kbins, weights=fourier_amplitudes)[0]
        / np.histogram(knrm, bins=kbins)[0]
    )
    
    # power_spectrum *= (4.0 / 3.0 * np.pi * (kbins[1:] ** 3 - kbins[:-1] ** 3))
    power_spectrum *= folding ** 3

    return kvals, power_spectrum / np.prod(deposition.shape)
