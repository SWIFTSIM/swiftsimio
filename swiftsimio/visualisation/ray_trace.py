"""
Ray tracing module for visualisation.
"""

# There should be three implementations here:
# - An example case that uses 'screening' with multiple panes.
# - An example case that builds a 3D mesh and uses that
# - An example case that uses a 'real' algorithm (even if it has to
#   be single-threaded)

from typing import Union
import numpy as np
import math

from swiftsimio.objects import cosmo_array
from swiftsimio.reader import __SWIFTParticleDataset, SWIFTDataset
from swiftsimio.visualisation.projection_backends.kernels import kernel_gamma, kernel

from swiftismio.accelerated import jit
import unyt

@jit(nopython=True, fastmath=True)
def core_panels(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    h: np.float32,
    m: np.float32,
    res: int,
    panels: int,
    min_z: np.float64,
    max_z: np.float64,
) -> np.array:
    """
    Creates a 2D array of the projected density of particles in a 3D volume using the
    'renormalised' strategy, with multiple panels across the z-range.

    Parameters
    ----------

    x: np.array[np.float64]
        The x-coordinates of the particles.

    y: np.array[np.float64]
        The y-coordinates of the particles.

    z: np.array[np.float64]
        The z-coordinates of the particles.

    h: np.array[np.float32]
        The smoothing lengths of the particles.

    m: np.array[np.float32]
        The masses of the particles.

    res: int
        The resolution of the output array.

    panels: int
        The number of panels to use in the z-direction.

    min_z: np.float64
        The minimum z-coordinate of the volume.

    max_z: np.float64
        The maximum z-coordinate of the volume.

    Returns
    -------

    A 3D array of shape (res, res, panels) containing the projected density in each pixel.
    """
    output = np.zeros((res, res, panels))
    maximal_array_index = res - 1

    number_of_particles = len(x)
    float_res = float(res)
    pixel_width = 1.0 / float_res

    assert len(y) == number_of_particles
    assert len(z) == number_of_particles
    assert len(h) == number_of_particles
    assert len(m) == number_of_particles

    z_per_panel = (max_z - min_z) / panels

    inverse_cell_area = float_res * float_res

    for i in range(number_of_particles):
        panel = z[i] // z_per_panel

        if panel < 0 or panel >= panels:
            continue

        particle_cell_x = int(float_res * x[i])
        particle_cell_y = int(float_res * y[i])

        kernel_width = kernel_gamma * h[i]
        cells_spanned = int(1.0 + kernel_width * float_res)

        if (
            particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # Can happily skip this particle
            continue

        if cells_spanned <= 1:
            output[particle_cell_x, particle_cell_y, panel] += m[i] * inverse_cell_area
            continue

        normalisation = 0.0

        for cell_x in range(
            particle_cell_x - cells_spanned, particle_cell_x + cells_spanned + 1
        ):
            distance_x = (float(cell_x) + 0.5) * pixel_width - x[i]
            distance_x_2 = distance_x * distance_x

            for cell_y in range(
                particle_cell_y - cells_spanned, particle_cell_y + cells_spanned + 1
            ):
                distance_y = (float(cell_y) + 0.5) * pixel_width - y[i]
                distance_y_2 = distance_y * distance_y

                r = math.sqrt(distance_x_2 + distance_y_2)

                normalisation += kernel(r / kernel_width)

        # Now have the normalisation
        normalisation = m[i] * inverse_cell_area / normalisation

        for cell_x in range(
            # Ensure that the lowest x value is 0, otherwise we'll segfault
            max(0, particle_cell_x - cells_spanned),
            # Ensure that the highest x value lies within the array bounds,
            # otherwise we'll segfault (oops).
            min(particle_cell_x + cells_spanned + 1, maximal_array_index + 1),
        ):
            distance_x = (float(cell_x) + 0.5) * pixel_width - x[i]
            distance_x_2 = distance_x * distance_x

            for cell_y in range(
                max(0, particle_cell_y - cells_spanned),
                min(particle_cell_y + cells_spanned + 1, maximal_array_index + 1),
            ):
                distance_y = (float(cell_y) + 0.5) * pixel_width - y[i]
                distance_y_2 = distance_y * distance_y

                r = math.sqrt(distance_x_2 + distance_y_2)

                output[cell_x, cell_y, panel] += (
                    kernel(r / kernel_width) * normalisation
                )

    return output


def panel_pixel_grid(
    data: __SWIFTParticleDataset,
    boxsize: unyt.unyt_array,
    resolution: int,
    panels: int,
    project: Union[str, None] = "masses",
    region: Union[None, unyt.unyt_array] = None,
    mask: Union[None, np.array] = None,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, unyt.unyt_array] = None,
) -> unyt.unyt_array:
    if rotation_center is not None:
        try:
            if rotation_center.units == data.coordinates.units:
                pass
            else:
                raise unyt.exceptions.InvalidUnitOperation(
                    "Units of coordinates and rotation center must agree"
                )
        except AttributeError:
            raise unyt.exceptions.InvalidUnitOperation(
                "Ensure that rotation_center is a unyt array with the same units as coordinates"
            )

    number_of_particles = data.coordinates.shape[0]

    if project is None:
        m = np.ones(number_of_particles, dtype=np.float32)
    else:
        m = getattr(data, project)
        if data.coordinates.comoving:
            if not m.compatible_with_comoving():
                raise AttributeError(
                    f'Physical quantity "{project}" is not compatible with comoving coordinates!'
                )
        else:
            if not m.compatible_with_physical():
                raise AttributeError(
                    f'Comoving quantity "{project}" is not compatible with physical coordinates!'
                )
        m = m.value 

     # This provides a default 'slice it all' mask.
    if mask is None:
        mask = np.s_[:]

    box_x, box_y, box_z = boxsize

    # Set the limits of the image.
    z_slice_included = False

    if region is not None:
        x_min, x_max, y_min, y_max = region[:4]

        if len(region) == 6:
            z_slice_included = True
            z_min, z_max = region[4:]
        else:
            z_min = unyt.unyt_quantity(0.0, units=box_z.units)
            z_max = box_z
    else:
        x_min = unyt.unyt_quantity(0.0, units=box_x.units)
        x_max = box_x
        y_min = unyt.unyt_quantity(0.0, units=box_y.units)
        y_max = box_y
        z_min = unyt.unyt_quantity(0.0, units=box_z.units)
        z_max = box_z


    x_range = x_max - x_min
    y_range = y_max - y_min

    # Deal with non-cubic boxes:
    # we always use the maximum of x_range and y_range to normalise the coordinates
    # empty pixels in the resulting square image are trimmed afterwards
    max_range = max(x_range, y_range)
    
    try:
        hsml = data.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.smoothing_length
    if data.coordinates.comoving:
        if not hsml.compatible_with_comoving():
            raise AttributeError(
                f"Physical smoothing length is not compatible with comoving coordinates!"
            )
    else:
        if not hsml.compatible_with_physical():
            raise AttributeError(
                f"Comoving smoothing length is not compatible with physical coordinates!"
            )

    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, z = np.matmul(rotation_matrix, (data.coordinates - rotation_center).T)

        x += rotation_center[0]
        y += rotation_center[1]
        z += rotation_center[2]
    else:
        x, y, z = data.coordinates.T

    return core_panels(
        x=x[mask],
        y=y[mask],
        z=z[mask],
        h=hsml[mask],
        m=m[mask],
        res=resolution,
        panels=panels,
        min_z=z_min,
        max_z=z_max,
    )

def panel_gas(
    data: SWIFTDataset,
    resolution: int,
    panels: int,
    project: Union[str, None] = "masses",
    region: Union[None, unyt.unyt_array] = None,
    mask: Union[None, np.array] = None,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, unyt.unyt_array] = None,
) -> cosmo_array:
    image = panel_pixel_grid(
        data=data.gas,
        boxsize=data.metadata.boxsize,
        resolution=resolution,
        panels=panels,
        project=project,
        region=region,
        mask=mask,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
    )

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        max_range = max(x_range, y_range)
        units = 1.0 / (max_range ** 2)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))
    else:
        max_range = max(data.metadata.boxsize[0], data.metadata.boxsize[1])
        units = 1.0 / (max_range ** 2)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 2)

    comoving = data.gas.coordinates.comoving
    coord_cosmo_factor = data.gas.coordinates.cosmo_factor
    if project is not None:
        units *= getattr(data.gas, project).units
        project_cosmo_factor = getattr(data.gas, project).cosmo_factor
        new_cosmo_factor = project_cosmo_factor / coord_cosmo_factor ** 2
    else:
        new_cosmo_factor = coord_cosmo_factor ** (-2)

    return cosmo_array(
        image, units=units, cosmo_factor=new_cosmo_factor, comoving=comoving
    )