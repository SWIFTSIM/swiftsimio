"""
Two-dimensional colour map support, along with example colour maps.
"""

from typing import List, Optional, Iterable

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

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

    norm_x = np.float32(len(map_grid) - 1)
    norm_y = np.float32(len(map_grid[0]) - 1)

    for index in range(number_of_values):
        horizontal = norm_x * min(max(first_values[index], 0.0), 1.0)
        vertical = norm_y * min(max(second_values[index], 0.0), 1.0)

        horizontal_base = np.int32(horizontal)
        vertical_base = np.int32(vertical)

        # Could do some more fancy interpolation but I'm sure this will do.
        output_values[index] = map_grid[horizontal_base, vertical_base]

    return output_values


class Cmap2D(object):
    """
    A generic two dimensional implementation of a colour map.
    
    Developer use only.
    """

    _color_map_grid: Optional[np.array] = None

    # Properties for color maps that generate
    colors: List[List[float]] = None
    coordinates: List[List[float]] = None

    def __init__(
        self, name: Optional[str] = None, description: Optional[str] = None,
    ):
        self.name = name
        self.description = description

        return

    def generate_color_map_grid(self):
        """
        Generates the colour map grid and stores it in
        ``_color_map_grid``. Imeplementation dependent.
        """

        self._color_map_grid = np.empty(
            (COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE), dtype=np.float32
        )

        return

    @property
    def color_map_grid(self):
        """
        Generates, or gets, the color map grid.
        """

        if self._color_map_grid is None:
            # Better make it!
            self.generate_color_map_grid()

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

        if include_points and self.colors is not None:
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
            vertical_values = np.ndarray(vertical_values)

        output_shape = horizontal_values.shape + (4,)

        values = apply_color_map(
            first_values=horizontal_values.flatten(),
            second_values=vertical_values.flatten(),
            map_grid=self.color_map_grid,
        )

        return values.reshape(output_shape)


class LinearSegmentedCmap2D(Cmap2D):
    """
    A two dimensional implementation of the linear segmented
    colour map.
    """

    def __init__(
        self,
        colors: List[List[float]],
        coordinates: List[List[float]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        
        colors: List[List[float]]
            Individual colors (at ``coordinates`` below) that make up
            the color map.
        
        coordinates: List[List[float]]
            2D coordinates in the plane to place the above ``colors``
            at.
            
        name: str, optional
            Name of this color map (metadata)
        
        description: str, optional
            Optional metadata description of this colour map.
            
        
        See Also
        --------
        
        ``LinearSegmentedCmap2DHSV``, a cousin of this class that
        combines colours using the HSV space rather than RGB used
        here.
        """
        super().__init__(name, description)

        self.colors = colors
        self.coordinates = coordinates

        return

    def generate_color_map_grid(self):
        """
        Generates the color map grid.
        """
        rgba_grid = np.zeros(
            COLOR_MAP_GRID_SIZE * COLOR_MAP_GRID_SIZE * 4, dtype=np.float32
        ).reshape((COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE, 4))

        values = np.linspace(0, 1, COLOR_MAP_GRID_SIZE)

        rgba_values = [ensure_rgba(color) for color in self.colors]

        for x_ind in range(COLOR_MAP_GRID_SIZE):
            for y_ind in range(COLOR_MAP_GRID_SIZE):
                weights = 0.0

                for rgba, coordinate in zip(rgba_values, self.coordinates):
                    dx = values[y_ind] - coordinate[0]
                    dy = values[x_ind] - coordinate[1]

                    r = np.sqrt(dx * dx + dy * dy)

                    weight = np.maximum(1.0 - r, 0.0)

                    rgba_grid[x_ind, y_ind] += rgba * weight
                    weights += weight

                rgba_grid[x_ind, y_ind] /= weights * 1.0001

        self._color_map_grid = rgba_grid

        return self._color_map_grid


class LinearSegmentedCmap2DHSV(Cmap2D):
    """
    A two dimensional implementation of the linear segmented
    colour map, using the HSV space to combine the colours.
    
    Parameters
    ----------
    
    colors: List[List[float]]
        Individual colors (at ``coordinates`` below) that make up
        the color map.
    
    coordinates: List[List[float]]
        2D coordinates in the plane to place the above ``colors``
        at.
        
    name: str, optional
        Name of this color map (metadata)
    
    description: str, optional
        Optional metadata description of this colour map.
        
    
    See Also
    --------
    
    ``LinearSegmentedCmap2D``, a cousin of this class that
    combines colours using the RGB space rather than HSV used
    here.
    """

    def __init__(
        self,
        colors: List[List[float]],
        coordinates: List[List[float]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(name, description)

        self.colors = colors
        self.coordinates = coordinates

        return

    def generate_color_map_grid(self):
        """
        Generates the color map grid.
        """
        hsv_grid = np.zeros(
            COLOR_MAP_GRID_SIZE * COLOR_MAP_GRID_SIZE * 3, dtype=np.float32
        ).reshape((COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE, 3))

        a_grid = np.zeros(
            COLOR_MAP_GRID_SIZE * COLOR_MAP_GRID_SIZE, dtype=np.float32
        ).reshape((COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE))

        values = np.linspace(0, 1, COLOR_MAP_GRID_SIZE)

        hsv_values = [
            np.array(rgb_to_hsv(ensure_rgba(color)[:-1])) for color in self.colors
        ]
        a_values = [ensure_rgba(color)[-1] for color in self.colors]

        for x_ind in range(COLOR_MAP_GRID_SIZE):
            for y_ind in range(COLOR_MAP_GRID_SIZE):
                weights = 0.0

                for hsv, a, coordinate in zip(hsv_values, a_values, self.coordinates):
                    dx = values[y_ind] - coordinate[0]
                    dy = values[x_ind] - coordinate[1]

                    r = np.sqrt(dx * dx + dy * dy)

                    weight = np.maximum(1.0 - r, 0.0)

                    hsv_grid[x_ind, y_ind] += hsv * weight
                    a_grid[x_ind, y_ind] += a * weight

                    weights += weight

                hsv_grid[x_ind, y_ind] /= weights * 1.0001
                a_grid[x_ind, y_ind] /= weights * 1.0001

        self._color_map_grid = np.empty(
            COLOR_MAP_GRID_SIZE * COLOR_MAP_GRID_SIZE * 4, dtype=np.float32
        ).reshape((COLOR_MAP_GRID_SIZE, COLOR_MAP_GRID_SIZE, 4))

        self._color_map_grid[:, :, :-1] = hsv_to_rgb(hsv_grid)
        self._color_map_grid[:, :, -1] = a_grid

        return self._color_map_grid


class ImageCmap2D(Cmap2D):
    """
    Creates a 2D color map from an image loaded from disk.
    """

    def __init__(
        self,
        filename: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        
        file_path: str
            Path to the image to use as the color map.
            
        name: str, optional
            Name of this color map (metadata)
            
        description: str, optional
            Optional metadata description of this colour map.
        """

        super().__init__(name=name, description=description)

        self.filename = filename

        return

    def generate_color_map_grid(self):
        """
        Loads the image from file and stores it as the internal
        array.
        """

        try:
            from PIL import Image
        except:
            raise ImportError(
                "Unable to import pillow, which must be installed "
                "to use color maps generated from images."
            )

        self._color_map_grid = (
            np.array(Image.open(self.filename)).astype(np.float32) / 255.0
        )

        return


# Define built in color maps.

bower = LinearSegmentedCmap2D(
    colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    name="bower",
)
