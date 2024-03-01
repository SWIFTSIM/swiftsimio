from swiftsimio.visualisation.smoothing_length.sph import get_hsml as sph_get_hsml
from swiftsimio.visualisation.smoothing_length.nearest_neighbours import (
    get_hsml as nearest_neighbours_get_hsml,
)
from swiftsimio.visualisation.smoothing_length.generate import (
    generate_smoothing_lengths,
)


backends_get_hsml = {
    "sph": sph_get_hsml,
    "nearest_neighbours": nearest_neighbours_get_hsml,
}
