SWIFTsimIO
==========

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
