"""
Py-SPHViewer integration for SWIFTsimIO.

This allows you to hook directly into the Scene and QuickView objects,
and provides helpful wrappers around the functions in py-sphviewer.
"""

import numpy as np

from swiftsimio.reader import __SWIFTParticleDataset
from swiftsimio.objects import cosmo_array
from swiftsimio.optional_packages import TREE_AVAILABLE, SPHVIEWER_AVAILABLE, viewer
from unyt import unyt_array

from typing import Union, List

from .smoothing_length_generation import generate_smoothing_lengths


class SPHViewerWrapper(object):
    """
    Wrapper for Py-SPHViewer to use SWIFTsimIO data structures.
    """

    # Forward declarations
    # Internal smoothing lengths that are used in the case where we
    # need to generate our own
    _internal_smoothing_lengths = None
    # Pixel grid output units
    smooth_units = None

    def __init__(
        self,
        dataset,
        smooth_over: Union[unyt_array, cosmo_array, str] = "masses",
        hsml_name: Union[str, None] = "smoothing_lengths",
    ):
        """
        Initialise the Particles class of py-sphviewer. Takes three arguments:

        + data, the particle dataset (e.g. data.gas would be render the gas)
        + hsml_name, the name of the object that contains smoothing lengths. If this
                     is None, we will attempt to create smoothing lengths that
                     encompass 32 nearest neighbours.
        + smooth_over, the name of the object to smooth over. This defaults to
                       masses, such that we return the projected mass density. This
                       can also be an arbritary unyt or cosmo array.
        
        Then, we can use any data available in that object to render the system.
        """

        if not SPHVIEWER_AVAILABLE:
            raise ImportError("Unable to find py-sphviewer on your system")

        self.data = dataset

        if isinstance(smooth_over, unyt_array) or isinstance(smooth_over, cosmo_array):
            self.smooth_over = smooth_over
        elif isinstance(smooth_over, str):
            self.smooth_over = getattr(self.data, smooth_over)
        else:
            raise AttributeError(
                "Invalid type {} passed to smooth_over parameter. "
                "Only allowed classes are: str, unyt_array, cosmo_array.".format(
                    type(smooth_over)
                )
            )

        self.__set_smoothing_lengths(hsml_name)
        self.__create_particles_instance()

        return

    def __set_smoothing_lengths(self, hsml_name: Union[str, None]):
        """
        Internal function for setting smoothing lengths. If None, then we 
        continue to create the smoothing lengths using an internal tree
        structure.
        """

        # Parameters required to generate smoothing lengths
        number_of_neighbours = int(
            round(self.data.metadata.hydro_scheme["Kernel target N_ngb"][0])
        )
        kernel_eta = self.data.metadata.hydro_scheme["Kernel eta"][0]

        kernel_gamma = ((3.0 * number_of_neighbours) / (4.0 * 3.14159)) ** (
            1 / 3
        ) / kernel_eta

        if hsml_name is None:
            self._internal_smoothing_lengths = generate_smoothing_lengths(
                self.data.coordinates,
                boxsize=self.data.metadata.boxsize,
                kernel_gamma=kernel_gamma,
            )
        else:
            try:
                self._internal_smoothing_lengths = getattr(self.data, hsml_name)
            except AttributeError:
                self._internal_smoothing_lengths = generate_smoothing_lengths(
                    self.data.coordinates,
                    boxsize=self.data.metadata.boxsize,
                    kernel_gamma=kernel_gamma,
                )

    def __create_particles_instance(self):
        """
        Internal function for creating the particles instance.
        
        Requires the setting of the smoothing lengths first.
        """

        if self._internal_smoothing_lengths is None:
            raise AssertionError(
                "Property _internal_smoothing_lengths should have been set already."
                "Please report to developers on GitHub."
            )

        if self._internal_smoothing_lengths.units != self.data.coordinates.units:
            raise AssertionError(
                "Smoothing lengths and coordinates are not provided in the same units! "
                "To use py-sphviewer integration, please set these to being equal "
                "by using .convert_to_units() on both arrays."
            )
        else:
            self.length_units = self.data.coordinates.units

        self.smooth_units = self.smooth_over.units / (self.length_units ** 2)

        self.particles = viewer.Particles(
            pos=self.data.coordinates.value,
            hsml=self._internal_smoothing_lengths.value,
            mass=self.smooth_over.value,
        )

        return

    def get_autocamera(self):
        """
        Sets a sensible value for the camera based on the camera's built in
        properties.
        """

        self.camera = viewer.Camera()
        self.camera.set_autocamera(self.particles)

        return self.camera

    def get_camera(
        self,
        x: Union[None, float] = None,
        y: Union[None, float] = None,
        z: Union[None, float] = None,
        r: Union[None, float] = None,
        t: Union[None, float] = None,
        p: Union[None, float] = None,
        zoom: Union[None, float] = None,
        roll: Union[None, float] = None,
        xsize: Union[None, int] = None,
        ysize: Union[None, int] = None,
        extent: Union[None, List[float]] = None,
    ):
        """
        Get the py-sphviewer camera object. This also sets it as self.camera that is used later.

        Properties are:

        + x, y, z: Cartesian co-ordinates of the object you're looking at
        + r: Cartesian distance to the object
        + t: ?????????? TODO
        + p: ?????????? TODO
        + zoom: ?????????? TODO
        + roll: ?????????? TODO
        + xsize, ysize: Pixel size of your output
        + extent: Area to render between
        """

        def convert_if_not_none(parameter):
            # Convert our parameter to the length units of the rest of
            # the data if it is not None or some non-standard choice.
            if parameter is None:
                return None
            elif isinstance(parameter, str):
                # This is the case if r="infinity"
                return parameter
            else:
                return parameter.to(self.length_units).value

        self.camera = viewer.Camera(
            x=convert_if_not_none(x),
            y=convert_if_not_none(y),
            z=convert_if_not_none(z),
            r=convert_if_not_none(r),
            t=t,
            p=p,
            zoom=zoom,
            roll=roll,
            xsize=xsize,
            ysize=ysize,
            extent=extent,
        )

        return self.camera

    def get_scene(self, camera: Union["viewer.Camera", None] = None):
        """
        Get the scene for a given camera. If there is no camera provided,
        we use the internal self.camera. If this is not set, then we raise
        an AttributeError.
        """

        if camera is not None:
            self.scene = viewer.Scene(self.particles, camera)
        elif self.camera is not None:
            self.scene = viewer.Scene(self.particles, self.camera)
        else:
            raise AttributeError(
                "You must make a choice for the camera, either by calling "
                ".get_camera() or by defining your own py-sphviewer camera "
                "and passing it to the get_scene function."
            )

        return self.scene

    def get_render(self):
        """
        Returns the render object (and sets self.render) using the internal
        scene object.

        We also provide .image and .extent as values that represent the render's
        image and extent including the input units.
        """

        self.render = viewer.Render(self.scene)

        self.image = self.render.get_image() * self.smooth_units
        self.extent = self.render.get_extent() * self.length_units

        return self.render

    def quick_view(
        self, xsize: int, ysize: int, r: Union[None, float] = None, **kwargs
    ):
        """
        Analogous to sphviewer.tools.QuickView but easier to directly call.
        Note that here we do not logscale any of the quantities.

        Here you must call 
        """

        self.get_autocamera()
        self.camera.set_params(xsize=xsize, ysize=ysize, r=r, **kwargs)

        self.get_scene()

        return self.get_render()
