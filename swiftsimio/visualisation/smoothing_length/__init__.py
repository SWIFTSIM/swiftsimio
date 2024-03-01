from swiftsimio.visualisation.smoothing_length.sph import get_hsml as sph_get_hsml
from swiftsimio.visualisation.smoothing_length.nearest_neighbors import (
    get_hsml as nearest_neighbors_get_hsml,
)
from swiftsimio.visualisation.smoothing_length.generate import (
    generate_smoothing_lengths,
)


backends_get_hsml = {
    "sph": sph_get_hsml,
    "nearest_neighbors": nearest_neighbors_get_hsml,
}
