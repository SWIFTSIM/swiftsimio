"""Functions for reading a subset of a SWIFT dataset and writing it to a new file."""

from swiftsimio.masks import SWIFTMask
from swiftsimio.accelerated import read_ranges_from_file
from swiftsimio._file_utils import is_dataset, is_soft_link
import swiftsimio.metadata as metadata

import h5py
import numpy as np


def get_swift_name(name: str) -> str:
    """
    Return the particle type name used in SWIFT.

    Parameters
    ----------
    name : str
        Swiftsimio particle name (e.g. gas).

    Returns
    -------
    str
        SWIFT particle type corresponding to `name` (e.g. PartType0).
    """
    part_type_names = [
        k for k, v in metadata.particle_types.particle_name_underscores.items()
    ]
    part_types = [
        v for k, v in metadata.particle_types.particle_name_underscores.items()
    ]
    return part_type_names[part_types.index(name)]


def get_dataset_mask(
    mask: SWIFTMask, dataset_name: str, suffix: str | None = None
) -> np.ndarray:
    """
    Return appropriate mask or mask size for given dataset.

    Parameters
    ----------
    mask : SWIFTMask
        The mask used to define subset that is written to new snapshot.
    dataset_name : str
        The name of the dataset we're interested in. This is the name from the
        hdf5 file (i.e. "PartType0", rather than "gas").
    suffix : str, optional
        Specify a suffix string to append to dataset underscore name to return
        something other than the dataset mask. This is specifically used for
        returning the mask size by setting suffix="_size", which would return,
        for example mask.gas_size.

    Returns
    -------
    np.ndarray
        Mask for the appropriate dataset.
    """
    suffix = "" if suffix is None else suffix

    if mask.metadata.shared_cell_counts:
        return getattr(mask, f"_shared{suffix}", None)
    elif "PartType" in dataset_name:
        part_type = dataset_name.lstrip("/").split("/")[0]
        mask_name = metadata.particle_types.particle_name_underscores[part_type]
        return getattr(mask, f"{mask_name}{suffix}", None)
    else:
        return None


def find_datasets(
    input_file: h5py.File,
    dataset_names: list[str] = [],
    path: str | None = None,
    recurse: bool = False,
) -> list[str]:
    """
    Recursively finds all the datasets in the snapshot and writes them to a list.

    Parameters
    ----------
    input_file : h5py.File
        HDF5 file handle for snapshot.

    dataset_names : list of str, optional
        Names of datasets found in the snapshot.

    path : str, optional
        The path to the current location in the snapshot.

    recurse : bool, optional
        Flag to indicate whether we're recursing or not.

    Returns
    -------
    list of str
        Names of datasets in ``path`` in ``input_file``.
    """
    if not recurse:
        dataset_names = []

    if path is not None:
        keys = input_file[path].keys()
    else:
        keys = input_file.keys()
        path = ""

    for key in keys:
        subpath = f"{path}/{key}"
        if is_dataset(input_file[subpath]):
            dataset_names.append(subpath)
        elif input_file[subpath].keys() is not None:
            find_datasets(input_file, dataset_names, subpath, recurse=True)

    return dataset_names


def find_links(
    input_file: h5py.File,
    link_names: list = [],
    link_paths: list = [],
    path: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Recursively finds all the links in the snapshot and writes them to a list.

    Parameters
    ----------
    input_file : h5py.File
        HDF5 file handle for snapshot.

    link_names : list of str
        Names of links found in the snapshot.

    link_paths : list of str
        Paths where links found in the snapshot point to.

    path : str, optional
        The path to the current location in the snapshot.

    Returns
    -------
    list of str
        List of the names in ``input_file``.

    list of str
        List of the paths in ``input_file``.
    """
    if path is not None:
        keys = input_file[path].keys()
    else:
        keys = input_file.keys()
        path = ""

    link_names = []
    link_paths = []
    for key in keys:
        subpath = f"{path}/{key}"
        dataset = input_file.get(subpath, getlink=True)
        if is_soft_link(dataset):
            link_names.append(subpath.lstrip("/"))
            link_paths.append(dataset.path)
        else:
            try:
                if input_file[subpath].keys() is not None:
                    find_links(input_file, link_names, link_paths, subpath)
            except AttributeError:
                # it's a Dataset (so has no `keys` attribute), skip
                pass

    return link_names, link_paths


def update_metadata_counts(
    infile: h5py.File, outfile: h5py.File, mask: SWIFTMask
) -> None:
    """
    Recalculate the cell particle counts and offsets from particles present in the subset.

    Parameters
    ----------
    infile : h5py.File
        File handle for input snapshot.

    outfile : h5py.File
        File handle for output subset of snapshot.

    mask : SWIFTMask
        The mask being used to define subset.
    """
    offsets_path = (
        "Cells/OffsetsInFile" if "Cells/OffsetsInFile" in infile else "Cells/Offsets"
    )
    outfile.create_group("Cells")
    outfile.create_group("Cells/Counts")
    outfile.create_group(offsets_path)

    # Get the particle counts and offsets in the cells
    particle_counts, particle_offsets = mask.get_masked_counts_offsets()

    # Loop over each particle type in the cells and update their counts
    counts_dsets = find_datasets(infile, path="/Cells/Counts")
    for part_type in particle_counts:
        for dset in counts_dsets:
            if mask.metadata.shared_cell_counts:
                outfile[dset] = particle_counts[part_type]
            elif get_swift_name(part_type) in dset:
                outfile[dset] = particle_counts[part_type]

    # Loop over each particle type in the cells and update their offsets
    offsets_dsets = find_datasets(infile, path=offsets_path)
    for part_type in particle_offsets:
        for dset in offsets_dsets:
            if mask.metadata.shared_cell_counts:
                outfile[dset] = particle_offsets[part_type]
            elif get_swift_name(part_type) in dset:
                outfile[dset] = particle_offsets[part_type]

    # Copy the cell centres and metadata
    infile.copy("/Cells/Centres", outfile, name="/Cells/Centres")
    outfile["/Cells/Centres"][...] = outfile["/Cells/Centres"][...][mask.cell_sort,]
    infile.copy("/Cells/Meta-data", outfile, name="/Cells/Meta-data")
    if (
        "MinPositions" in infile["/Cells"].keys()
        and "MaxPositions" in infile["/Cells"].keys()
    ):
        infile.copy("/Cells/MinPositions", outfile, name="/Cells/MinPositions")
        infile.copy("/Cells/MaxPositions", outfile, name="/Cells/MaxPositions")
        for k, v in outfile["/Cells/MinPositions"].items():
            outfile[f"/Cells/MinPositions/{k}"][...] = v[...][mask.cell_sort,]
        for k, v in outfile["/Cells/MaxPositions"].items():
            outfile[f"/Cells/MaxPositions/{k}"][...] = v[...][mask.cell_sort,]


def write_metadata(
    infile: h5py.File, outfile: h5py.File, links_list: list[str], mask: SWIFTMask
) -> None:
    """
    Copy over all the metadata from snapshot to output file.

    Parameters
    ----------
    infile : h5py.File
        HDF5 file handle for input snapshot.

    outfile : h5py.File
        HDF5 file handle for output snapshot.

    links_list : list of str
        Names of links found in the snapshot.

    mask : SWIFTMask
        The mask being used to define subset.
    """
    update_metadata_counts(infile, outfile, mask)

    skip_list = links_list.copy()
    skip_list += ["Cells"]
    skip_list += set(group.split("/")[0] for group in mask.metadata.present_groups)
    for field in infile.keys():
        if not any([substr for substr in skip_list if substr in field]):
            # HDF5<14 can segfault for these groups when infile.copy() is called
            # due to the arrays of strings stored in the attributes
            if field in ["Header", "Parameters"]:
                header = outfile.create_group(field)
                for k, v in infile[field].attrs.items():
                    header.attrs[k] = v
            else:
                infile.copy(field, outfile)


def write_datasubset(
    infile: h5py.File,
    outfile: h5py.File,
    mask: SWIFTMask,
    dataset_names: list[str],
    links_list: list[str],
) -> None:
    """
    Write subset of all datasets contained in snapshot according to specified mask.

    Parameters
    ----------
    infile : h5py.File
        HDF5 file handle for input snapshot.

    outfile : h5py.File
        HDF5 file handle for output snapshot.

    mask : SWIFTMask
        The mask used to define subset that is written to new snapshot.

    dataset_names : list of str
        Names of datasets found in the snapshot.

    links_list : list of str
        Names of links found in the snapshot.
    """
    skip_list = links_list.copy()
    skip_list.extend(["Cells", "SubgridScheme", "PartTypeNames"])
    if mask is not None:
        for name in dataset_names:
            if any([substr for substr in skip_list if substr in name]):
                continue

            # get output dtype and size
            first_value = infile[name][0]
            output_type = first_value.dtype
            output_size = first_value.size
            mask_size = get_dataset_mask(mask, name, suffix="_size")
            if output_size != 1:
                output_shape = (mask_size, output_size)
            else:
                output_shape = mask_size

            dataset_mask = get_dataset_mask(mask, name)

            subset = read_ranges_from_file(
                infile[name],
                dataset_mask,
                output_shape=output_shape,
                output_type=output_type,
            )

            # Write the subset
            outfile.create_dataset(name, data=subset)
            for attr_name, attr_value in infile[name].attrs.items():
                outfile[name].attrs.create(attr_name, attr_value)


def connect_links(
    outfile: h5py.File, links_list: list[str], paths_list: list[str]
) -> None:
    """
    Connect up the links to the appropriate path.

    Parameters
    ----------
    outfile : h5py.File
        File containing the hdf5 subsnapshot.

    links_list : list of str
        List of names of soft links.

    paths_list : list of str
        List of paths specifying how to link each soft link.
    """
    for i in range(len(links_list)):
        outfile[links_list[i]] = h5py.SoftLink(paths_list[i])


def write_subset(output_file: str, mask: SWIFTMask) -> None:
    """
    Write subset of snapshot according to specified mask to new snapshot file.

    Parameters
    ----------
    output_file : str
        Path to input snapshot.

    output_file : str
        Path to output snapshot.

    mask : SWIFTMask
        The mask used to define subset that is written to new snapshot.
    """
    # Open the files
    with mask.metadata.open_file() as infile, h5py.File(output_file, "w") as outfile:
        # Write metadata and data subset
        list_of_links, list_of_link_paths = find_links(infile)
        write_metadata(infile, outfile, list_of_links, mask)
        write_datasubset(infile, outfile, mask, find_datasets(infile), list_of_links)
        connect_links(outfile, list_of_links, list_of_link_paths)
