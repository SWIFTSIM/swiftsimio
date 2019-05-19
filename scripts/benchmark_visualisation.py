"""
A short benchmarking script. As of 19th May 2019,
our visualisation takes approximately 21 seconds to
complete this task on a 2018 Macbook Pro with a
2.7 GhZ i7.
"""

from swiftsimio.visualisation import scatter
from numpy import ones_like, array, float32, zeros
from numpy.random import rand, seed
from time import time

number_of_particles = 100_000
res = 1024

seed(1234)

print("Generating particles")
x = rand(number_of_particles).astype(float32)
y = rand(number_of_particles).astype(float32)
h = rand(number_of_particles).astype(float32) * 0.2
m = ones_like(h)
print("Finished generating particles")

print("Compiling")
t = time()
scatter(
    array([0.0], dtype=float32),
    array([0.0], dtype=float32),
    array([1.0], dtype=float32),
    array([0.01], dtype=float32),
    128,
)
print(f"Took {time() - t} to compile.")

print("Scattering")
t = time()
image = scatter(x, y, m, h, res)
dt_us = time() - t
print(f"Took {dt_us} to scatter.")

try:
    from sphviewer.tools import QuickView
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print("Comparing with pySPHViewer")
    coordinates = zeros((number_of_particles, 3))
    coordinates[:, 0] = x
    coordinates[:, 1] = y
    h = 1.778_002 * h  # The kernel_gamma we use.

    t = time()
    qv = QuickView(
        coordinates,
        hsml=h,
        mass=m,
        x_size=res,
        y_size=res,
        r="infinity",
        plot=False,
        logscale=False,
    ).get_image()
    dt_pysphviewer = time() - t
    print(f"pySPHViewer took {dt_pysphviewer} on the same problem.")
    print(f"Note that pySPHViewer is running in single-threaded mode.")

    ratio = dt_us / dt_pysphviewer

    if ratio < 1.0:
        print(f"That makes us {1 / ratio} x faster 😀 ")
    else:
        print(f"That makes pySphviewer {ratio} x faster 😔")

except:
    print("pySPHViewer not available for comparison")
