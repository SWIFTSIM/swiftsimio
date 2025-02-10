import numpy as np


def _get_projection_field(data, field_name):
    if field_name is None:
        m = np.ones(data.particle_ids.size)
    else:
        m = getattr(data, field_name)
        if data.coordinates.comoving:
            if not m.compatible_with_comoving():
                raise AttributeError(
                    f'Physical quantity "{field_name}" is not compatible with comoving '
                    "coordinates!"
                )
        else:
            if not m.compatible_with_physical():
                raise AttributeError(
                    f'Comoving quantity "{field_name}" is not compatible with physical '
                    "coordinates!"
                )
        m = m.value  # slated for removal
    return m


def _get_region_limits(data, region, z_slice=None, require_cubic=False):
    z_slice_included = z_slice is not None
    if not z_slice_included:
        z_slice = np.zeros_like(data.metadata.boxsize[0])
    box_x, box_y, box_z = data.metadata.boxsize
    if region is not None:
        x_min, x_max, y_min, y_max = region[:4]
        if len(region) == 6:
            z_slice_included = True
            z_min, z_max = region[4:]
        else:
            z_min, z_max = np.zeros_like(box_z), box_z
    else:
        x_min, x_max = np.zeros_like(box_x), box_x
        y_min, y_max = np.zeros_like(box_y), box_y
        z_min, z_max = np.zeros_like(box_z), box_z

    if z_slice_included and (z_slice > box_z) or (z_slice < np.zeros_like(box_z)):
        raise ValueError("Please enter a slice value inside the box.")

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = np.max([x_range, y_range])

    if require_cubic and not (
        np.isclose(x_range, y_range) and np.isclose(x_range, z_range)
    ):
        raise AttributeError(
            "Projection code is currently not able to handle non-cubic images"
        )

    return (
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        x_range,
        y_range,
        z_range,
        max_range,
    )


def _get_rotated_coordinates(data, rotation_matrix, rotation_center):
    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, z = np.matmul(rotation_matrix, (data.coordinates - rotation_center).T)

        x += rotation_center[0]
        y += rotation_center[1]
        z += rotation_center[2]
    else:
        x, y, z = data.coordinates.T
    return x, y, z
