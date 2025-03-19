"""
Tools for creating power spectra from SWIFT data.
"""

from numpy import float32, float64, int32, zeros, ndarray, zeros_like
import numpy as np
import scipy.fft

from swiftsimio.optional_packages import tqdm
from swiftsimio.accelerated import jit, NUM_THREADS, prange
from swiftsimio import cosmo_array, cosmo_quantity
from swiftsimio.reader import __SWIFTGroupDataset

from typing import Optional, Dict, Tuple


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
        floating-point version (i.e. 2.0^folding).
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
        floating-point version (i.e. 2.0^folding).
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
    data: __SWIFTGroupDataset,
    resolution: int,
    project: str = "masses",
    folding: int = 0,
    parallel: bool = False,
) -> cosmo_array:
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
    cosmo_array[float32, float32, float32]
        The deposition.
    """

    # Get the positions and masses
    folding = 2.0 ** folding
    positions = data.coordinates
    quantity = getattr(data, project)

    if positions.comoving:
        if not quantity.compatible_with_comoving():
            raise AttributeError(
                f'Physical quantity "{project}" is not compatible with comoving '
                "coordinates!"
            )
    else:
        if not quantity.compatible_with_physical():
            raise AttributeError(
                f'Comoving quantity "{project}" is not compatible with physical '
                "coordinates!"
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

    units = 1.0 / (
        data.metadata.boxsize[0] * data.metadata.boxsize[1] * data.metadata.boxsize[2]
    )
    units.convert_to_units(1.0 / data.metadata.boxsize.units ** 3)

    units *= quantity.units
    new_cosmo_factor = quantity.cosmo_factor / (coord_cosmo_factor ** 3)

    return cosmo_array(
        deposition, comoving=comoving, cosmo_factor=new_cosmo_factor, units=units
    )


def folded_depositions_to_power_spectrum(
    depositions: Dict[int, cosmo_array],
    box_size: cosmo_array,
    number_of_wavenumber_bins: int,
    cross_depositions: Optional[Dict[int, cosmo_array]] = None,
    wavenumber_range: Optional[Tuple[cosmo_quantity]] = None,
    log_wavenumber_bins: bool = True,
    workers: Optional[int] = None,
    minimal_sample_modes: Optional[int] = 0,
    cutoff_above_wavenumber_fraction: Optional[float] = None,
    track_progress: bool = False,
    transition: str = "simple",
    shot_noise_norm: Optional[float] = None,
) -> Tuple[cosmo_array]:
    """
    Convert some folded depositions to power spectra.

    Parameters
    ----------

    depositions: dict[int, cosmo_array]
        Dictionary of depositions, where the key is the base folding.
        So that would be 0 for no folding, 1 for a half-box-size folding,
        2 for a quarter, etc. The 'real' folding is 2 ** depositions.keys().
    box_size: cosmo_array
        The box size of the deposition, from the dataset.
    number_of_wavenumber_bins: int
        The number of bins to use in the power spectrum.
    cross_depositions: Optional[dict[int, cosmo_array]]
        An optional dictionary of cross-depositions, where the key is the folding.
    wavenumber_range: Optional[tuple[cosmo_quantity]]
        The range of wavenumbers to use. Officially optional, but is required for
        now.
    log_wavenumber_bins: bool
        Whether to use logarithmic bins. By default true.
    workers: Optional[int]
        The number of threads to use.
    minimal_sample_modes: Optional[int]
        The minimum number of modes to sample from each indivudal
        power spectrum. Useful to cut down on small-scale sample noise
        completely.
    cutoff_above_wavenumber_fraction: Optional[float]
        Cut off the individual spectra at this fraction of their
        maximally sampled wavenumber. Ignored for the last fold.
    track_progress: bool = False
        Whether to display a progress bar representing each fold.
        Requires the tqdm package to be installed.
    transition: str
        How to transition between different folds.
        "simple" means use a simple scheme where the highest number
        of modes is used. "average" uses a weighted averaging
        scheme.
    shot_noise_norm: Optional[float]
        The normalization to apply to the shot noise. Usually the number of
        particles or galaxies used to create the mesh.

    Returns
    -------

    wavenumber_bins: cosmo_array[float32]
        The wavenumber bins.

    wavenumber_centers: cosmo_array[float32]
        The centers of the wavenumber bins.

    power_spectrum: cosmo_array[float32]
        The power spectrum.

    folding_tracker: np.array
        A tracker of the contribution of various folded elements.
    """

    if cross_depositions is not None:
        if not set(depositions.keys()) == set(cross_depositions.keys()):
            raise ValueError(
                "Depositions and cross depositions need to have the same foldings."
            )

    if wavenumber_range is None:
        raise NotImplementedError

    if log_wavenumber_bins:
        wavenumber_bins = np.geomspace(
            np.min(cosmo_array(wavenumber_range)),
            np.max(cosmo_array(wavenumber_range)),
            number_of_wavenumber_bins + 1,
        )

    else:
        wavenumber_bins = np.linspace(
            np.min(cosmo_array(wavenumber_range)),
            np.max(cosmo_array(wavenumber_range)),
            number_of_wavenumber_bins + 1,
        )
    wavenumber_bins.name = "Wavenumber bins"

    wavenumber_centers = 0.5 * (wavenumber_bins[1:] + wavenumber_bins[:-1])
    wavenumber_centers.name = r"Wavenumbers $k$"
    box_volume = np.prod(box_size)
    power_spectrum = cosmo_array(
        np.zeros(number_of_wavenumber_bins),
        units=box_volume.units,
        comoving=box_volume.comoving,
        cosmo_factor=box_volume.cosmo_factor ** -1,
        name="Power spectrum $P(k)$",
    )
    folding_tracker = np.ones(number_of_wavenumber_bins, dtype=float)
    contributed_counts = np.zeros(number_of_wavenumber_bins, dtype=int)
    corrected_wavenumber_centers = wavenumber_centers.copy()
    first_folding = min(depositions.keys())
    final_folding = max(depositions.keys())

    iterator = sorted(list(depositions.keys()))

    if track_progress:
        iterator = tqdm(iterator, desc="Processing folds")

    for folding in iterator:
        (
            folded_wavenumber_centers,
            folded_power_spectrum,
            folded_counts,
        ) = deposition_to_power_spectrum(
            deposition=depositions[folding],
            box_size=box_size,
            folding=folding,
            wavenumber_bins=wavenumber_bins,
            workers=workers,
            shot_noise_norm=shot_noise_norm,
        )

        use_bins = folded_counts > (
            minimal_sample_modes if folding != first_folding else 0
        )

        # Even if we do not have a specified cutoff, we should still not allow
        # any weird noise make us take bins above our theoretical best, though
        # for the last one we don't really care so much.

        if folding != final_folding:
            cutoff_wavenumber = (
                2.0 ** folding * np.min(depositions[folding].shape) / np.min(box_size)
            )

            if cutoff_above_wavenumber_fraction is not None:
                maximally_sampled_wavenumber = np.max(
                    folded_wavenumber_centers[use_bins]
                )
                cutoff_wavenumber = np.min(
                    cutoff_above_wavenumber_fraction * maximally_sampled_wavenumber,
                    cutoff_above_wavenumber_fraction * cutoff_wavenumber,
                )

            use_bins = np.logical_and(
                use_bins, folded_wavenumber_centers < cutoff_wavenumber
            )

        if transition == "simple":
            # Simple scheme. Highest number of counts (i.e. best 'resolved')
            # bins wins.
            prefer_bins = np.logical_and(folded_counts > contributed_counts, use_bins)

            power_spectrum[prefer_bins] = folded_power_spectrum[prefer_bins]
            corrected_wavenumber_centers[prefer_bins] = folded_wavenumber_centers[
                prefer_bins
            ].to(corrected_wavenumber_centers.units)
            folding_tracker[prefer_bins] = 2.0 ** folding

            contributed_counts[prefer_bins] = folded_counts[prefer_bins]
        elif transition == "average":
            # This more complex averaging scheme is left in here for prosperity, but
            # shouldn't be used as it underestimates power on large scales.
            # We should use the simple scheme instead.
            raise ValueError("The average scheme is not supported. Use simple.")

            # Our more complex averaging scheme.
            # Cbrt gives you the 'linear' number of included bins.
            new_weight = np.cbrt(folded_counts)
            existing_weight = np.cbrt(contributed_counts)

            # Smoothly transition between folds, prioritizing better-sampled
            # bins in newer (or older!) folds.
            transition_norm = np.maximum(new_weight + existing_weight, 1)

            power_spectrum[use_bins] = (
                (power_spectrum * existing_weight + new_weight * folded_power_spectrum)
                / transition_norm
            )[use_bins]

            corrected_wavenumber_centers[use_bins] = (
                (
                    corrected_wavenumber_centers * existing_weight
                    + folded_wavenumber_centers * new_weight
                )
                / transition_norm
            )[use_bins].to(corrected_wavenumber_centers.units)

            # For debugging, we calculate an effective fold number.
            folding_tracker[use_bins] = (
                (folding_tracker * existing_weight + (2.0 ** folding) * new_weight)
                / transition_norm
            )[use_bins]

            contributed_counts[use_bins] += folded_counts[use_bins]
        else:
            raise ValueError("Unacceptable transition scheme.")

    return (
        wavenumber_bins,
        corrected_wavenumber_centers,
        power_spectrum,
        folding_tracker,
    )


def deposition_to_power_spectrum(
    deposition: cosmo_array,
    box_size: cosmo_array,
    folding: int = 0,
    cross_deposition: Optional[cosmo_array] = None,
    wavenumber_bins: Optional[cosmo_array] = None,
    workers: Optional[int] = None,
    shot_noise_norm: Optional[float] = None,
) -> Tuple[cosmo_array]:
    """
    Convert a deposition to a power spectrum, by default
    using a linear binning strategy.

    Parameters
    ----------
    deposition: ~swiftsimio.objects.cosmo_array[float32, float32, float32]
        The deposition to convert to a power spectrum.
    box_size: ~swiftsimio.objects.cosmo_array
        The box size of the deposition, from the dataset.
    folding: int
        The folding number (i.e. box-size is divided by 2^folding)
        that was used here.
    cross_deposition: ~swiftsimio.objects.cosmo_array[float32, float32, float32]
        An optional second deposition to cross-correlate with the first.
        If not provided, we assume you want an auto-spectrum.
    wavenumber_bins: ~swiftsimio.objects.cosmo_array[float32], optional
        Optionally you can provide the specific bins that you would like to use.
    workers: Optional[int]
        The number of threads to use.
    shot_noise_norm: Optional[float]
        The normalization to apply to the shot noise. Usually the number of
        particles or galaxies used to create the mesh.

    Returns
    -------

    wavenumber_centers: ~swiftsimio.objects.cosmo_array[float32]
        The k-values of the power spectrum, with units. These are the
        real bin centers, calculated from the mean value of k that was
        used in the binning process.

    power_spectrum: ~swiftsimio.objects.cosmo_array[float32]
        The power spectrum, with units.

    binned_counts: np.array[int32]
        Bin counts for the power spectrum; useful for shot noise.

    Notes
    -----

    This is adapted from Bert Vandenbroucke's code at
    https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    """

    if not len(deposition.shape) == 3:
        raise ValueError("Deposition must be a 3D array")

    if not deposition.shape[0] == deposition.shape[1] == deposition.shape[2]:
        raise ValueError("Deposition must be a cube")

    if cross_deposition is not None:
        assert (
            deposition.shape == cross_deposition.shape
        ), "Depositions must have the same shape"

    folding = 2.0 ** folding

    box_size_folded = box_size[0] / folding
    npix = deposition.shape[0]

    mean_deposition = np.mean(deposition)
    overdensity = (deposition - mean_deposition) / mean_deposition

    fft = scipy.fft.fftn(overdensity / np.prod(deposition.shape), workers=workers)

    if cross_deposition is not None:
        mean_cross_deposition = np.mean(cross_deposition)
        cross_overdensity = (
            cross_deposition - mean_cross_deposition
        ) / mean_cross_deposition

        conj_fft = scipy.fft.fftn(
            cross_overdensity / np.prod(deposition.shape), workers=workers
        ).conj()
    else:
        conj_fft = fft.conj()

    fourier_amplitudes = (fft * conj_fft).real * box_size_folded ** 3

    # Calculate k-value spacing (centered FFT)
    dk = 2 * np.pi / (box_size_folded)

    # Create k-values array (adjust range based on your needs)
    kfreq = np.fft.fftfreq(npix, d=1 / dk) * npix
    kfreq3d = np.meshgrid(kfreq, kfreq, kfreq)
    knrm = np.sqrt(kfreq3d[0] ** 2 + kfreq3d[1] ** 2 + kfreq3d[2] ** 2)

    # knrm = knrm.flatten()
    # fourier_amplitudes = fourier_amplitudes.flatten()

    if wavenumber_bins is None:
        kbins = np.arange(0.5, npix // 2 + 1, 1.0) * dk
    else:
        kbins = wavenumber_bins

    binned_amplitudes = np.histogram(knrm, bins=kbins, weights=fourier_amplitudes)[0]
    binned_counts = np.histogram(knrm, bins=kbins)[0]
    # Also compute the 'real' average wavenumber point contributing to this bin.
    binned_wavenumbers = np.histogram(knrm, bins=kbins, weights=knrm)[0]

    zero_mask = binned_counts == 0

    # Avoid divide by zero
    binned_amplitudes[zero_mask] = 0.0

    divisor = binned_counts.copy()
    divisor[zero_mask] = 1

    # Correct for folding
    binned_amplitudes *= folding ** 3

    # Correct units and names
    wavenumbers = binned_wavenumbers / divisor
    wavenumbers.name = "Wavenumber $k$"

    shot_noise = (
        (box_size[0] ** 3 / shot_noise_norm)
        if shot_noise_norm is not None
        else zeros_like(box_size[0] ** 3)  # copy cosmo properties
    )
    power_spectrum = (binned_amplitudes / divisor) - shot_noise

    power_spectrum.name = "Power Spectrum $P(k)$"

    return wavenumbers, power_spectrum, binned_counts
