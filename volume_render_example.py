# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from swiftsimio import load
from swiftsimio.visualisation import volume_render

# Load the data
data = load("test_data/eagle_6.hdf5")

# Rough location of an interesting galaxy in the volume.
region = [
    0.225 * data.metadata.boxsize[0],
    0.275 * data.metadata.boxsize[0],
    0.12 * data.metadata.boxsize[1],
    0.17 * data.metadata.boxsize[1],
    0.45 * data.metadata.boxsize[2],
    0.5 * data.metadata.boxsize[2],
]

# Render the volume (note 1024 is reasonably high resolution so this won't complete
# immediately; you should consider using 256, etc. for testing).
rendered = volume_render.render_gas(data, resolution=1024, region=region, parallel=True)

# Quick view! By projecting along the final axis you can get
# the projected density from the rendered image.
plt.imsave("volume_render_quick_view.png", LogNorm()(rendered.sum(-1)))

# Now we will move onto the real volume rendering. Let's use the log of the density;
# using the real density leads to low contrast images.
log_rendered = np.log10(rendered)

# The volume rendering function expects centers of 'bins' and widths. These
# bins actually represent gaussian functions around a specific density (or other
# visualization quantity). The brightest pixel value is at center. We will
# visualise this later!
width = 0.1
std = np.std(log_rendered)
mean = np.mean(log_rendered)

# It's helpful to choose the centers relative to the data you have. When making
# a movie, you will obviously want to choose the centers to be the same for each
# frame.
centers = [mean + x * std for x in [1.0, 3.0, 5.0, 7.0]]

# This will visualize your render options. The centers are shown as gaussians and
# vertical lines.
fig, ax = volume_render.visualise_render_options(
    centers=centers, widths=width, cmap="viridis"
)

histogram, edges = np.histogram(
    log_rendered.flat,
    bins=128,
    range=(min(centers) - 5.0 * width, max(centers) + 5.0 * width),
)
bc = (edges[:-1] + edges[1:]) / 2.0

# The normalization here is the height of a gaussian!
ax.plot(bc, histogram / (np.max(histogram) * np.sqrt(2.0 * np.pi) * width))
ax.semilogy()
ax.set_xlabel("$\\log_{10}(\\rho)$")

plt.savefig("volume_render_options.png")

# Now we can really visualize the rendering.
img, norms = volume_render.visualise_render(
    log_rendered,
    centers,
    widths=width,
    cmap="viridis",
)

# Sometimes, these images can be a bit dark. You can increase the brightness using
# tools like PIL or in your favourite image editor.
from PIL import Image, ImageEnhance

pilimg = Image.fromarray((img * 255.0).astype(np.uint8))
enhanced = ImageEnhance.Contrast(ImageEnhance.Brightness(pilimg).enhance(2.0)).enhance(
    1.2
)

enhanced.save("volume_render_example.png")
