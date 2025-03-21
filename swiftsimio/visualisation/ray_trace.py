"""
Ray tracing module for visualisation.
"""

from typing import Union
import numpy as np

from swiftsimio.objects import cosmo_array
from swiftsimio.reader import __SWIFTGroupDataset, SWIFTDataset

from swiftsimio.visualisation.ray_trace_backends import backends
from swiftsimio.visualisation.smoothing_length import backends_get_hsml
from swiftsimio.visualisation._vistools import (
    _get_projection_field,
    _get_region_info,
    _get_rotated_coordinates,
    backend_restore_cosmo_and_units,
)


def panel_pixel_grid(
    data: __SWIFTGroupDataset,
    resolution: int,
    panels: int,
    project: Union[str, None] = "masses",
    region: Union[None, cosmo_array] = None,
    mask: Union[None, np.array] = None,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
) -> cosmo_array:

    m = _get_projection_field(data, project)
    region_info = _get_region_info(data, region)
    hsml = backends_get_hsml["sph"](data)
    x, y, z = _get_rotated_coordinates(data, rotation_matrix, rotation_center)
    mask = np.s_[...] if mask is None else mask

    # There's a parallel version of core_panels but it seems
    # that it's never used anywhere.
    norm = region_info["x_range"] * region_info["y_range"]
    return backend_restore_cosmo_and_units(backends["core_panels"], norm=norm)(
        x=x[mask] / region_info["max_range"],
        y=y[mask] / region_info["max_range"],
        z=z[mask],
        h=hsml[mask] / region_info["max_range"],
        m=m[mask],
        res=resolution,
        panels=panels,
        min_z=region_info["z_min"],
        max_z=region_info["z_max"],
    )


def panel_gas(
    data: SWIFTDataset,
    resolution: int,
    panels: int,
    project: Union[str, None] = "masses",
    region: Union[None, cosmo_array] = None,
    mask: Union[None, np.array] = None,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
) -> cosmo_array:
    return panel_pixel_grid(
        data=data.gas,
        resolution=resolution,
        panels=panels,
        project=project,
        region=region,
        mask=mask,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
    )
