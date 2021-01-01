"""
Two-dimensional colour map support, along with example colour maps.
"""

from typing import List, Optional, Iterable

import numpy as np

from swiftsimio.accelerated import jit

COLOR_MAP_GRID_SIZE = 256


def ensure_rgba(input_color: Iterable[float]) -> np.array:
    """
    Ensures a colour is RGBA compliant.

    Default alpha if missing: 1.0.

    Parameters
    ----------

    input_color: iterable
        An iterable of maximum length 4, with RGBA values
        encoded as floating point 0.0 -> 1.0.


    Returns
    -------

    array_color: np.array
        An array of length 4 as an RGBA color.

    """

    array_color = np.zeros(4, dtype=np.float32)
    array_color[-1] = 1.0

    for index, rgba in enumerate(input_color):
        array_color[index] = rgba

    return array_color


@jit(nopython=True, fastmath=True)
def apply_color_map(first_values, second_values, map_grid):
    """
    Applies a 2D colour map by providing a 2D linear interpolation
    to the known fixed grid points. Not to be called on its own,
    as the map itself is provided by the ``LinearSegmentedCmap2D``,
    but this is provided separately so it can be ``numba``-accelerated.
    
    Parameters
    ----------
    
    first_values: iterable[float]
        Array or list to loop over, containing floats ranging from 0.0
        to 1.0. Provides the normalisation for the horizontal
        component. Must be one-dimensional.
        
    second_values: iterable[float]
        Array or list to loop over, containing floats ranging from 0.0
        to 1.0. Provides the normalisation for the vertical
        component. Must be one-dimensional.
        
    map_grid: np.ndarray
        2D numpy array proided by ``LinearSegmentedCmap2D``.

    
    Returns
    -------
    
    np.ndarray
        An N by 4 array (where N is the length of ``first_value`` and
        ``second_value``) of RGBA components.
    """

    number_of_values = len(first_values)
    output_values = np.empty(number_of_values * 4, dtype=np.float32).reshape(
        (number_of_values, 4)
    )

    for index in range(number_of_values):
        horizontal = COLOR_MAP_GRID_SIZE * min(max(first_values[index], 0.0), 1.0)
        vertical = COLOR_MAP_GRID_SIZE * min(max(second_values[index], 0.0), 1.0)

        horizontal_base = np.int32(horizontal)
        vertical_base = np.int32(vertical)

        # Could do some more fancy interpolation but I'm sure this will do.
        output_values[index] = map_grid[horizontal_base, vertical_base]

    return output_values


class LinearSegmentedCmap2D(object):
    """
    A two dimensional implementation of the linear segmented
    colour map.
    """

    _color_map_grid: Optional[np.array] = None

    def __init__(
        self,
        colors: List[List[float]],
        coordinates: List[List[float]],
        name: Optional[str] = None,
    ):
        self.colors = colors
        self.coordinates = coordinates
        self.name = name

        return

    @property
    def color_map_grid(self):
        """
        Generates, or gets, the color map grid.
        """

        if self._color_map_grid is None:
            # Better make it!
            self._color_map_grid = np.zeros(
                COLOR_MAP_GRID_SIZE * COLOR_MAP_GRID_SIZE * 4, dtype=np.float32
            ).reshape((COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE, 4))

            # Set initial alpha values to 1.0
            self._color_map_grid[:, :, 3] = 1.0

            x, y = np.meshgrid(*[np.linspace(0, 1, COLOR_MAP_GRID_SIZE)] * 2)

            # Need to loop through each colour, giving the grid each component
            # in turn.
            for color, coordinate in zip(self.colors, self.coordinates):
                array_color = ensure_rgba(color)

                dx = x - coordinate[0]
                dy = y - coordinate[1]

                r = np.sqrt(dx * dx + dy * dy)

                weights = np.maximum(1.0 - r, 0.0)

                for x_ind in range(COLOR_MAP_GRID_SIZE):
                    for y_ind in range(COLOR_MAP_GRID_SIZE):
                        # Screen blend mode
                        self._color_map_grid[x_ind, y_ind, :] = 1.0 - (
                            1.0 - self._color_map_grid[x_ind, y_ind, :]
                        ) * (1.0 - weights[x_ind, y_ind] * array_color)

        return self._color_map_grid

    def plot(self, ax, include_points: bool = False):
        """
        Plot the color map on axes.
        
        Parameters
        ----------
        
        ax: matplotlib.Axis
            Axis to be plotted on.
            
        include_points: bool, optional
            If true, plot the individual colours as points that make
            up the color map. Default: False.
        """

        ax.imshow(self.color_map_grid, origin="lower", extent=[0, 1, 0, 1])
        ax.set_xlabel("Horizontal Value (first index)")
        ax.set_ylabel("Vertical Value (second index)")

        if include_points:
            for color, coordinate in zip(self.colors, self.coordinates):
                rgba_color = ensure_rgba(color)

                ax.scatter(*coordinate, color=rgba_color, edgecolor="white")

        return

    def __call__(self, horizontal_values, vertical_values):
        """
        Apply the 2D color map to some data. Both sets of values
        must be of the same shape.
        
        Parameters
        ----------
        
        horizontal_values: iterable
            Values for the first parameter in the color map
        
        vertical_values: iterable
            Values for the second parameter in the color map
            
        Returns
        -------
        
        mapped: np.ndarray
            RGBA array using the internal colour map.
        """

        if isinstance(horizontal_values, list) or isinstance(horizontal_values, tuple):
            horizontal_values = np.array(horizontal_values)

        if isinstance(vertical_values, list) or isinstance(vertical_values, tuple):
            vertical_values = np.array(vertical_values)

        values = apply_color_map(
            first_values=horizontal_values.flatten(),
            second_values=vertical_values.flatten(),
            map_grid=self.color_map_grid,
        )

        values.reshape(horizontal_values.shape + (4,))

        return values
