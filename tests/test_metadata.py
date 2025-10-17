"""
Tests some known good states with the metadata.
"""

from swiftsimio import metadata


def test_same_contents():
    """
    Tests that there are the same arrays in each of the following:

        + particle fields
        + unit fields
        + cosmology fields

    We treat the particle fields as the ground truth.
    """

    cosmology = metadata.cosmology_fields.generate_cosmology(1.0, 1.0)
    units = metadata.unit_fields.generate_units(1.0, 1.0, 1.0, 1.0, 1.0)
    particle = {x: getattr(metadata.particle_fields, x) for x in units.keys()}

    # Do we cover all the same particle fields?

    assert cosmology.keys() == particle.keys()
    assert units.keys() == particle.keys()

    for ptype in cosmology.keys():
        assert set(units[ptype].keys()) == set(particle[ptype].values())
        assert set(cosmology[ptype].keys()) == set(particle[ptype].values())

    return
