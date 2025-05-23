import numpy as np
from warnings import warn
from functools import reduce
from swiftsimio.objects import cosmo_array
from swiftsimio._array_functions import _copy_cosmo_array_attributes_if_present


def _get_projection_field(data, field_name):
    if field_name is None:
        return np.ones_like(data.particle_ids)
    return reduce(getattr, field_name.split("."), data)


def _get_region_info(data, region, require_cubic=False, periodic=True):
    boxsize = data.metadata.boxsize
    if region is not None:
        region = cosmo_array(region)
    if data.coordinates.comoving:
        boxsize.convert_to_comoving()
        if region is not None:
            region.convert_to_comoving()
    elif data.coordinates.comoving is False:  # compare to False in case None
        boxsize.convert_to_physical()
        if region is not None:
            region.convert_to_physical()
    box_x, box_y, box_z = boxsize
    if region is not None:
        x_min, x_max, y_min, y_max = region[:4]
        if len(region) == 6:
            region_includes_z = True
            z_min, z_max = region[4:]
        else:
            region_includes_z = False
            z_min, z_max = np.zeros_like(box_z), box_z
    else:
        region_includes_z = False
        x_min, x_max = np.zeros_like(box_x), box_x
        y_min, y_max = np.zeros_like(box_y), box_y
        z_min, z_max = np.zeros_like(box_z), box_z

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = np.r_[x_range, y_range].max()

    if require_cubic and not (
        np.isclose(x_range, y_range) and np.isclose(x_range, z_range)
    ):
        raise AttributeError(
            "Projection code is currently not able to handle non-cubic images."
        )

    periodic_box_x, periodic_box_y, periodic_box_z = (
        boxsize / max_range if periodic else np.zeros(3)
    )

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "max_range": max_range,
        "region_includes_z": region_includes_z,
        "periodic_box_x": periodic_box_x,
        "periodic_box_y": periodic_box_y,
        "periodic_box_z": periodic_box_z,
    }


def _get_rotated_and_wrapped_coordinates(
    data, rotation_matrix, rotation_center, periodic
):
    if rotation_center is not None and periodic is True:
        raise ValueError(
            "Rotation and periodic boundaries in visualisation are incompatible."
        )
    if rotation_center is not None:
        if data.coordinates.comoving:
            rotation_center = rotation_center.to_comoving()
        elif data.coordinates.comoving is False:
            rotation_center = rotation_center.to_physical()
        # Rotate co-ordinates as required
        coords = (
            np.matmul(rotation_matrix, (data.coordinates - rotation_center).T).T
            + rotation_center
        )
    else:
        coords = data.coordinates
    if periodic:
        coords %= data.metadata.boxsize
    return coords.T


def backend_restore_cosmo_and_units(backend_func, norm=1.0):
    def wrapper(*args, **kwargs):
        comoving = getattr(kwargs["m"], "comoving", None)
        if comoving is True:
            if kwargs["x"].comoving is False or kwargs["y"].comoving is False:
                warn(
                    "Projecting a comoving quantity with physical input for coordinates. "
                    "Converting coordinate grid to comoving."
                )
            kwargs["x"].convert_to_comoving()
            kwargs["y"].convert_to_comoving()
            if kwargs["h"].comoving is False:
                warn(
                    "Projecting a comoving quantity with physical input for smoothing "
                    "lengths. Converting smoothing lengths to comoving."
                )
            kwargs["h"].convert_to_comoving()
            norm.convert_to_comoving()
        elif comoving is False:  # don't use else in case None
            if kwargs["x"].comoving or kwargs["y"].comoving:
                warn(
                    "Projecting a physical quantity with comoving input for coordinates. "
                    "Converting coordinate grid to physical."
                )
            kwargs["x"].convert_to_physical()
            kwargs["y"].convert_to_physical()
            if kwargs["h"].comoving:
                warn(
                    "Projecting a physical quantity with comoving input for smoothing "
                    "lengths. Converting smoothing lengths to physical."
                )
            kwargs["h"].convert_to_physical()
            norm.convert_to_physical()
        return (
            _copy_cosmo_array_attributes_if_present(
                kwargs["m"],
                backend_func(*args, **kwargs).view(cosmo_array),
                copy_units=True,
            )
            / norm
        )

    return wrapper
