SWIFTsimIO
==========

[![Build Status](https://travis-ci.com/SWIFTSIM/swiftsimio.svg?branch=master)](https://travis-ci.com/SWIFTSIM/swiftsimio)

The SWIFT astrophysical simulation code (http://swift.dur.ac.uk) is used widely. There
exists many ways of reading the data from SWIFT, which outputs HDF5 files. These range
from reading directly using `h5py` to using a complex system such as `yt`; however these
either are unsatisfactory (e.g. a lack of unit information in reading HDF5), or too
complex for most use-cases. This (thin) wrapper provides an object-oriented API to read
(dynamically) data from SWIFT.


Requirements
------------

This requires `python3.6.0` or higher. No effort will be made to support python versions
below this. Please update your systems.

### Python packages

+ `h5py`
+ `unyt`


Usage
-----

Example usage is shown below, which plots a density-temperature phase
diagram, with density and temperature given in CGS units:

```python
import swiftsimio as sw

# This loads all metadata but explicitly does _not_ read any particle data yet
data = sw.load("/path/to/swift/output")

import matplotlib.pyplot as plt

data.gas.density.convert_to_cgs()
data.gas.temperature.convert_to_cgs()

plt.loglog()

plt.scatter(
    data.gas.density,
    data.gas.temperature,
    s=1
)

plt.xlabel(fr"Gas density $\left[{data.gas.density.units.latex_repr}\right]$")
plt.ylabel(fr"Gas temperature $\left[{data.gas.temperature.units.latex_repr}\right]$")

plt.tight_layout()

plt.savefig("test_plot.png", dpi=300)
```

In the above it's important to note the following:

+ All metadata is read in when the `load` function is called.
+ Only the density and temperature (corresponding to the `PartType0/Density` and
  `PartType0/Temperature`) datasets are read in.
+ That data is only read in once the `convert_to_cgs` method is called.
+ `convert_to_cgs` converts data in-place; i.e. it returns `None`.
+ The data is cached and not re-read in when `plt.scatter` is called.


Writing datasets
----------------

Writing datasets that are valid for consumption for cosmological codes can be
difficult, especially when considering how to best use units. SWIFT uses a different
set of internal units (specified in your parameter file) that does not necessarily need
to be the same set of units that initial conditions are specified in. Nevertheless,
it is important to ensure that units in the initial conditions are all _consistent_
with each other. To facilitate this, we use `unyt` arrays. The below example generates
randomly placed gas particles with uniform densities.

```python
from swiftsimio import Writer
from swiftsimio.units import cosmo_units

import unyt
import numpy as np

# Box is 100 Mpc
boxsize = 100 * unyt.Mpc

# Generate object. cosmo_units corresponds to default Gadget-oid units
# of 10^10 Msun, Mpc, and km/s
x = Writer(cosmo_units, boxsize)

# 32^3 particles.
n_p = 32**3

# Randomly spaced coordinates from 0, 100 Mpc in each direction
x.gas.coordinates = np.random.rand(n_p, 3) * (100 * unyt.Mpc)

# Random velocities from 0 to 1 km/s
x.gas.velocities = np.random.rand(n_p, 3) * (unyt.km / unyt.s)

# Generate uniform masses as 10^6 solar masses for each particle
x.gas.masses = np.ones(n_p, dtype=float) * (1e6 * unyt.msun)

# Generate internal energy corresponding to 10^4 K
x.gas.internal_energy = np.ones(n_p, dtype=float) * (1e4 * unyt.kb * unyt.K) / (1e6 * unyt.msun)

# Generate initial guess for smoothing lengths based on MIPS
x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

# If IDs are not present, this automatically generates    
x.write("test.hdf5")
```

Then, running `h5glance` on the resulting `test.hdf5` produces:
```
test.hdf5
├Header
│ └5 attributes:
│   ├BoxSize: 100.0
│   ├Dimension: array [int64: 1]
│   ├Flag_Entropy_ICs: 0
│   ├NumPart_Total: array [int64: 6]
│   └NumPart_Total_HighWord: array [int64: 6]
├PartType0
│ ├Coordinates  [float64: 32768 × 3]
│ ├InternalEnergy       [float64: 32768]
│ ├Masses       [float64: 32768]
│ ├ParticleIDs  [float64: 32768]
│ ├SmoothingLength      [float64: 32768]
│ └Velocities   [float64: 32768 × 3]
└Units
  └5 attributes:
    ├Unit current in cgs (U_I): array [float64: 1]
    ├Unit length in cgs (U_L): array [float64: 1]
    ├Unit mass in cgs (U_M): array [float64: 1]
    ├Unit temperature in cgs (U_T): array [float64: 1]
    └Unit time in cgs (U_t): array [float64: 1]
```

**Note** you do need to be careful that your choice of unit system does
_not_ allow values over 2^31, i.e. you need to ensure that your
provided values (with units) when _written_ to the file are safe to 
be interpreted as (single-precision) floats. The only exception to
this is coordinates which are stored in double precision.


Ahead-of-time Masking
---------------------

SWIFT snapshots contain cell metadata that allow us to spatially mask the
data ahead of time. `swiftsimio` provides a number of objects that help with
this. See the example below.

```python
import swiftsimio as sw

# This creates and sets up the masking object.
mask = sw.mask("/path/to/swift/snapshot")

# This ahead-of-time creates a spatial mask based on the cell metadata.
mask.constrain_spatial([[0.2 * mask.metadata.boxsize[0], 0.7 * mask.metadata.boxsize[0]], None, None])

# Now, just for fun, we also constrain the density between 0.4 g/cm^3 and 0.8. This reads in
# the relevant data in the region, and tests it element-by-element.
density_units = mask.units.mass / mask.units.length**3
mask.constrain_mask("gas", "density", 0.4 * density_units, 0.8 * density_units)

# Now we can grab the actual data object. This includes the mask as a parameter.
data = sw.load("/Users/josh/Documents/swiftsim-add-anarchy/examples/SodShock_3D/sodShock_0001.hdf5", mask=mask)
```

When the attributes of this data object are accessed, transparently _only_ the ones that 
belong to the masked region (in both density and spatial) are read. I.e. if I ask for the
temperature of particles, it will recieve an array containing temperatures of particles
that lie in the region [0.2, 0.7] and have a density between 0.4 and 0.8 g/cm^3.


User-defined particle types
---------------------------

It is now possible to add user-defined particle types using generator functions through an
API in the `load()` function:

```python
import swiftsimio as sw
import swiftsimio.metadata.particle as swp
import swiftsimio.metadata.writer.required_fields as swmw
import swiftsimio.metadata.unit.unit_fields as swuf
import swiftsimio.metadata.cosmology as swcf
from swiftsimio.objects import cosmo_factor, a

def cosmo_factory(a_dependence):
    return cosmo_factor(a_dependence, scale_factor)


def generate_units(m, l, t, I, T):
    """
    Generates the units for a non-standard scheme.

    Needs to return a dictionary of dictionaries:

    {
        "particle_type_name" : {
            "field_name": m * m / l * T,
            ...
        },
        ...
    }
    """

    return ...


def generate_cosmology(scale_factor: float, gamma: float):
    """
    Generates the cosmology a-factors for a non-standard scheme.

    Needs to return a dictionary of dictionaries:

    {
        "particle_type_name" : {
            "field_name": cosmo_factory(a*a*a),
            ...
        },
        ...
    }
    """

    return ...


swp.particle_name_underscores[6] = "extratype"
swp.particle_name_class[6] = "Extratype"
swp.particle_name_text[6] = "Extratype"

swmw.extratype = {"smoothing_length": "SmoothingLength", **swmw.shared}

sw.metadata.particle_fields.extratype = {**sw.metadata.particle_fields.gas}

data = sw.load(
    "extra_test.hdf5",
    generate_unit_func=generate_units,
    generate_cosmology_func=generate_cosmology,
)
```

This feature is not ideal and will likely be changed in the future.

Image Creation
--------------

`swiftsimio` provides some very basic visualisation software that is designed
to create projections of the entire box along the z-axis, with x and y representing
the final square. Note that it only currently supports square boxes. These are
accelerated with `numba`, and we do not suggest using these routines unless you have it
installed.

You can do the following:

```python
from swiftsimio import load
from swiftsimio.visualisation import project_gas

data = load("my_snapshot_0000.hdf5")
# This creates a grid that has units K / Mpc^2, and can be transformed like
# any other unyt quantity
temperature_map = project_gas(data, resolution=1024, project="temperature")

from matplotlib.pyplot import imsave
from matplotlib.colors import LogNorm

# There is huge dynamic range usually in temperature, so log normalize it before
# saving.
imsave("temperature_map.png", LogNorm()(temperature_map.value), cmap="twilight")
```

It's also possible to create videos in a fairly straightforward way:
```python
from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

# project_gas_pixel_grid does not perform the unit calculation.

from p_tqdm import p_map

def load_and_make_image(number):
    filename = f"snapshot_{number:04d}.hdf5"
    data = load(filename)
    image = project_gas_pixel_grid(data, 1024, "temperature")

    return image

# Make frames in parallel (reading also parallel!)
frames = p_map(load_and_make_image, range(0, 120))

#... You can do your funcAnimation stuff here now you have the frames.
```


It is also possible to create _slice_ plots, rather than projections.
```
from swiftsimio import load
from swiftsimio.visualisation import slice_gas

data = load("my_snapshot_0000.hdf5")
# This creates a grid that has units K / Mpc^3, and can be transformed like
# any other unyt quantity. The slice variable gives where we want to slice through
# as a function of the box-size, so here we slice through the centre of the box.
temperature_map = slice_gas(data, slice=0.5, resolution=1024, project="temperature")

from matplotlib.pyplot import imsave
from matplotlib.colors import LogNorm

# There is huge dynamic range usually in temperature, so log normalize it before
# saving.
imsave("temperature_map.png", LogNorm()(temperature_map.value), cmap="twilight")
```