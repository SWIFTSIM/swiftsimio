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

from matplotlib.pyplot import imsave

imsave("test_image.png", image)
