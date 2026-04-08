"""Helper functions to read and parse the metadata attached to individual datasets."""

from swiftsimio.objects import cosmo_factor
from swiftsimio.metadata.objects import (
    SWIFTUnits,
    SWIFTMetadata,
)

import h5py
import unyt


def load_field_units(
    field_attributes: h5py.AttributeManager, unit_metadata: SWIFTUnits
) -> unyt.Unit:
    """
    Get the units from the HDF5 attribute for a field.

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    unit_metadata : ~swiftsimio.metadata.object.SWIFTUnits
        The metadata describing the unit system for the dataset.

    Returns
    -------
    unyt.Unit
        The loaded units.
    """
    unit_dict = {
        "I": unit_metadata.current,
        "L": unit_metadata.length,
        "M": unit_metadata.mass,
        "T": unit_metadata.temperature,
        "t": unit_metadata.time,
    }
    units = unyt.dimensionless
    for unit_type, unit in unit_dict.items():
        # We store the 'unit exponent' in the SWIFT metadata. This corresponds to the
        # power that we need to raise each unit to, to construct the units for this field.
        unit_exponent = field_attributes[f"U_{unit_type} exponent"][0]
        # Check if the exponent is 0 manually because of float precision
        if unit_exponent != 0.0:
            units = units * unit**unit_exponent
    # We can simplify like this because e.g.:
    # dimensionless / dimensionless == dimensionless
    # and dimensionless * Mpc / dimensionless == Mpc
    return units / unyt.dimensionless


def load_field_description(field_attributes: h5py.AttributeManager) -> str:
    """
    Get the text description from the HDF5 attribute for a field.

    A description of the mask is included if present (e.g. for SOAP filetypes).

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    Returns
    -------
    str
        The field description.
    """
    description = field_attributes.get("Description", "No description available.")
    if hasattr(description, "decode"):
        description = description.decode("utf-8")
    mask_str = (
        (
            "Only computed for objects where "
            f"{' + '.join(field_attributes['Mask Datasets'])} "
            f">= {field_attributes['Mask Threshold']}."
        )
        if field_attributes.get("Masked", False)
        else "Not masked."
    )
    return f"{description} {mask_str}"


def load_field_compression(field_attributes: h5py.AttributeManager) -> str:
    """
    Get the description of the compression filters from the HDF5 attribute for a field.

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    Returns
    -------
    str
        The compression filter description.
    """
    if not field_attributes.get("Is Compressed", True):
        return "Not compressed."
    comp = field_attributes.get(
        "Lossy compression filter", "No compression info available"
    )
    if hasattr(comp, "decode"):
        comp = comp.decode("utf-8")
    return comp


def load_field_cosmo_factor(
    field_attributes: h5py.AttributeManager, metadata: SWIFTMetadata
) -> cosmo_factor:
    """
    Construct the :class:`~swiftsimio.objects.cosmo_factor` for a field.

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    metadata : ~swiftsimio.metadata.objects.SWIFTMetadata
        The metadata for the SWIFT dataset.

    Returns
    -------
    ~swiftsimio.objects.cosmo_factor
        The cosmology information for the field.
    """
    return cosmo_factor.create(
        metadata.scale_factor, field_attributes.get("a-scale exponent", [0.0])[0]
    )


def load_field_physical(field_attributes: h5py.AttributeManager) -> bool:
    """
    Get the flag for comoving/physical coordinates from the HDF5 attribute for a field.

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    Returns
    -------
    bool
        The flag for comoving or physical coordinates.
    """
    return field_attributes.get("Value stored as physical", [0])[0] == 1


def load_field_valid_transform(field_attributes: h5py.AttributeManager) -> bool:
    """
    Get whether to allow comoving-physical conversion from the HDF5 attribute for a field.

    Parameters
    ----------
    field_attributes : ~h5py._hl.attrs.AttributeManager
        The :mod:`h5py` interface to the attributes for the field.

    Returns
    -------
    bool
        The flag for whether comoving-physical conversion is allowed.
    """
    # defaults to True for backwards compatibility
    return field_attributes.get("Property can be converted to comoving", [1])[0] == 1
