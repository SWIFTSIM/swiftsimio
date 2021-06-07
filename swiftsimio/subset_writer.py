"""
Contains functions for reading a subset of a SWIFT dataset and writing
it to a new file.
"""

from swiftsimio.reader import SWIFTUnits, SWIFTMetadata
from swiftsimio.masks import SWIFTMask
from swiftsimio.accelerated import read_ranges_from_file
import swiftsimio.metadata as metadata

import unyt
import h5py
import numpy as np
from typing import Optional, List


def get_swift_name(name: str) -> str:
    """
    Returns the particle type name used in SWIFT

    Parameters
    ----------
    name : str
        swiftsimio particle name (e.g. gas)

    Returns
    -------
    str
        SWIFT particle type corresponding to `name` (e.g. PartType0)
    """
    part_type_nums = [
        k for k, v in metadata.particle_types.particle_name_underscores.items()
    ]
    part_types = [
        v for k, v in metadata.particle_types.particle_name_underscores.items()
    ]
    part_type_num = part_type_nums[part_types.index(name)]
    return f"PartType{part_type_num}"


def get_dataset_mask(
    mask: SWIFTMask, dataset_name: str, suffix: Optional[str] = None
) -> np.ndarray:
    """
    Return appropriate mask or mask size for given dataset

    Parameters
    ----------
    mask : SWIFTMask
        the mask used to define subset that is written to new snapshot
    dataset_name : str
        the name of the dataset we're interested in. This is the name from the
        hdf5 file (i.e. "PartType0", rather than "gas")
    suffix : str, optional
        specify a suffix string to append to dataset underscore name to return
        something other than the dataset mask. This is specifically used for
        returning the mask size by setting suffix="_size", which would return,
        for example mask.gas_size

    Returns
    -------
    np.ndarray
        mask for the appropriate dataset

    """
    suffix = "" if suffix is None else suffix

    if "PartType" in dataset_name:
        part_type = [int(x) for x in filter(str.isdigit, dataset_name)][0]
        mask_name = metadata.particle_types.particle_name_underscores[part_type]
        return getattr(mask, f"{mask_name}{suffix}", None)
    else:
        return None


def find_datasets(
    input_file: h5py.File, dataset_names=[], path=None, recurse=False
) -> List[str]:
    """
    Recursively finds all the datasets in the snapshot and writes them to a list

    Parameters
    ----------
    input_file : h5py.File
        hdf5 file handle for snapshot
    dataset_names : list of str, optional
        names of datasets found in the snapshot
    path : str, optional
        the path to the current location in the snapshot
    recurse : bool, optional
        flag to indicate whether we're recursing or not

    Returns
    -------
    dataset_names : list of str
        names of datasets in `path` in `input_file`
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
        if isinstance(input_file[subpath], h5py.Dataset):
            dataset_names.append(subpath)
        elif input_file[subpath].keys() is not None:
            find_datasets(input_file, dataset_names, subpath, recurse=True)

    return dataset_names


def find_links(
    input_file: h5py.File,
    link_names: Optional[List] = [],
    link_paths: Optional[List] = [],
    path: Optional[str] = None,
) -> (List[str], List[str]):
    """
    Recursively finds all the links in the snapshot and writes them to a list

    Parameters
    ----------
    input_file : h5py.File
        hdf5 file handle for snapshot
    link_names : list of str, optional
        names of links found in the snapshot
    link_paths : list of str, optional
        paths where links found in the snapshot point to
    path : str, optional
        the path to the current location in the snapshot

    Returns
    -------
    link_names, link_paths : list of str, list of str
        lists of the names and links of paths in `input_file`
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
        if isinstance(dataset, h5py.SoftLink):
            link_names.append(subpath.lstrip("/"))
            link_paths.append(dataset.path)
        else:
            try:
                if input_file[subpath].keys() is not None:
                    find_links(input_file, link_names, link_paths, subpath)
            except:
                pass

    return link_names, link_paths


def update_metadata_counts(infile: h5py.File, outfile: h5py.File, mask: SWIFTMask):
    """
    Recalculates the cell particle counts and offsets based on the particles present in the subset

    Parameters
    ----------
    infile : h5py.File
        File handle for input snapshot
    outfile : h5py.File
        File handle for output subset of snapshot
    mask : SWIFTMask
        the mask being used to define subset
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
            if get_swift_name(part_type) in dset:
                outfile[dset] = particle_counts[part_type]

    # Loop over each particle type in the cells and update their offsets
    offsets_dsets = find_datasets(infile, path=offsets_path)
    for part_type in particle_offsets:
        for dset in offsets_dsets:
            if get_swift_name(part_type) in dset:
                outfile[dset] = particle_offsets[part_type]

    # Copy the cell centres and metadata
    infile.copy("/Cells/Centres", outfile, name="/Cells/Centres")
    infile.copy("/Cells/Meta-data", outfile, name="/Cells/Meta-data")


def write_metadata(
    infile: h5py.File, outfile: h5py.File, links_list: List[str], mask: SWIFTMask
):
    """
    Copy over all the metadata from snapshot to output file

    Parameters
    ----------
    infile : h5py.File
        hdf5 file handle for input snapshot
    outfile : h5py.File
        hdf5 file handle for output snapshot
    links_list : list of str
        names of links found in the snapshot
    mask : SWIFTMask
        the mask being used to define subset
    """

    update_metadata_counts(infile, outfile, mask)

    skip_list = links_list.copy()
    skip_list += ["PartType", "Cells"]
    for field in infile.keys():
        if not any([substr for substr in skip_list if substr in field]):
            infile.copy(field, outfile)


def write_datasubset(
    infile: h5py.File,
    outfile: h5py.File,
    mask: SWIFTMask,
    dataset_names: List[str],
    links_list: List[str],
):
    """
    Writes subset of all datasets contained in snapshot according to specified mask
    Parameters
    ----------
    infile : h5py.File
        hdf5 file handle for input snapshot
    outfile : h5py.File
        hdf5 file handle for output snapshot
    mask : SWIFTMask
        the mask used to define subset that is written to new snapshot
    dataset_names : list of str
        names of datasets found in the snapshot
    links_list : list of str
        names of links found in the snapshot
    """
    skip_list = links_list.copy()
    skip_list.extend(["Cells", "SubgridScheme"])
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


def connect_links(outfile: h5py.File, links_list: List[str], paths_list: List[str]):
    """
    Connects up the links to the appropriate path

    Parameters
    ----------
    outfile : h5py.File
        file containing the hdf5 subsnapshot
    links_list : list of str
        list of names of soft links
    paths_list : list of str
        list of paths specifying how to link each soft link
    """
    for i in range(len(links_list)):
        outfile[links_list[i]] = h5py.SoftLink(paths_list[i])


def write_subset(output_file: str, mask: SWIFTMask):
    """
    Writes subset of snapshot according to specified mask to new snapshot file

    Parameters
    ----------
    input_file : str
        path to input snapshot
    output_file : str
        path to output snapshot
    mask : SWIFTMask
        the mask used to define subset that is written to new snapshot
    """
    # Open the files
    infile = h5py.File(mask.metadata.filename, "r")
    outfile = h5py.File(output_file, "w")

    # Write metadata and data subset
    list_of_links, list_of_link_paths = find_links(infile)
    write_metadata(infile, outfile, list_of_links, mask)
    write_datasubset(infile, outfile, mask, find_datasets(infile), list_of_links)
    connect_links(outfile, list_of_links, list_of_link_paths)

    # Clean up
    infile.close()
    outfile.close()
