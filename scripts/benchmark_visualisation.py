"""
A short benchmarking script. As of 19th May 2019,
our visualisation takes approximately 21 seconds to
complete this task on a 2018 Macbook Pro with a
2.7 GhZ i7.
"""

from swiftsimio.visualisation import scatter
from numpy import ones_like, array, float32
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
print(f"Took {time() - t} to scatter.")

try:
    from sphviewer import QuickView

    print("Comparing with pySPHViewer")
    coordinates = np.zeros((number_of_particles, 3))
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
    print(f"pySPHViewer took {time() - t} on the same problem.")

except:
    print("pySPHViewer not available for comparison")
