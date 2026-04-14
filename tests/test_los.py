"""Tests for line-of-sight (LOS) files."""

import numpy as np
import pytest
import unyt

from swiftsimio import SWIFTLineOfSightMetadata, load, mask



def test_los_can_load(los_example):
    """Check that we can load a LOS file and access a standard dataset."""
    los = load(los_example)
    assert isinstance(los.metadata, SWIFTLineOfSightMetadata)
    assert los.los_0000.coordinates is not None


def test_los_group_attributes_cosmo_types_units_and_metadata(los_example):
    """LOS group attributes should be cosmology-aware objects with correct metadata."""
    los = load(los_example)
    group = los.los_0000
    a = float(los.metadata.scale_factor)

    for name in ("xpos", "ypos"):
        value = getattr(group, name)
        assert isinstance(value, cosmo_quantity)
        assert value.units == los.metadata.units.length.units
        assert value.comoving is True
        assert np.isclose(value.cosmo_factor.a_factor, a)

    for name in ("xaxis", "yaxis", "zaxis", "num_parts"):
        value = getattr(group, name)
        assert isinstance(value, cosmo_quantity)
        assert value.units == unyt.dimensionless
        assert value.comoving is False
        assert np.isclose(value.cosmo_factor.a_factor, 1.0)


def test_los_group_attribute_cached_after_first_access(los_example):
    """LOS group attributes should be lazy-loaded once and then cached."""
    los = load(los_example)
    assert los.los_0000._xpos is None
    los.los_0000.xpos
    assert los.los_0000._xpos is not None


def test_los_spatial_mask_not_supported(los_example):
    """LOS files do not provide cell metadata, so masks should fail."""
    with pytest.raises(NotImplementedError, match="Masking not supported"):
        mask(los_example)
