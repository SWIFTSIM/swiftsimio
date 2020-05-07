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


def get_dataset_mask(mask: SWIFTMask, dataset_name: str, suffix=""):
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
    if "PartType" in dataset_name:
        part_type = [int(x) for x in filter(str.isdigit, dataset_name)][0]
        mask_name = metadata.particle_types.particle_name_underscores[part_type]
        return getattr(mask, f"{mask_name}{suffix}", None)
    else:
        return None


def write_datasubset(infile, outfile, mask: SWIFTMask, dataset_names, links_list):
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


def write_metadata(infile, outfile, links_list, mask, restrict):
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
    """
    
    update_metadata_counts(infile, outfile, mask, restrict)

    skip_list = links_list.copy()
    skip_list += ["PartType", "Cells"]
    for field in infile.keys():
        if not any([substr for substr in skip_list if substr in field]):
            infile.copy(field, outfile)


def find_datasets(input_file: h5py.File, dataset_names=[], path=None, recurse = 0):
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
            find_datasets(input_file, dataset_names, subpath, recurse = 1)

    return dataset_names


def find_links(input_file: h5py.File, link_names=[], link_paths=[], path=None):
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
    """
    if path is not None:
        keys = input_file[path].keys()
    else:
        keys = input_file.keys()
        path = ""

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


def connect_links(outfile: h5py.File, links_list, paths_list):
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

def get_swift_name(name: str) -> str:
    part_type_nums = [k for k, v in metadata.particle_types.particle_name_underscores.items()]
    part_types = [v for k, v in metadata.particle_types.particle_name_underscores.items()]
    part_type_num = part_type_nums[part_types.index(name)]
    return f"PartType{part_type_num}"

def update_metadata_counts(infile: h5py.File, outfile: h5py.File, mask: SWIFTMask, restrict: np.ndarray):
    outfile.create_group("Cells")
    outfile.create_group("Cells/Counts")
    outfile.create_group("Cells/Offsets")

    # Get the particle counts and offsets in the cells
    particle_counts, particle_offsets = mask.refine_metadata_mask(restrict)

    # Loop over each particle type in the cells and update their counts
    counts_dsets = find_datasets(infile, path = "/Cells/Counts")
    for part_type in particle_counts:
        for dset in counts_dsets:
            if get_swift_name(part_type) in dset:
                outfile[dset] = particle_counts[part_type]
    
    # Loop over each particle type in the cells and update their offsets
    offsets_dsets = find_datasets(infile, path = "/Cells/Offsets")
    for part_type in particle_offsets:
        for dset in offsets_dsets:
            if get_swift_name(part_type) in dset:
                outfile[dset] = particle_offsets[part_type]

    # Copy the cell centres and metadata
    infile.copy("/Cells/Centres", outfile)
    infile.copy("/Cells/Meta-data", outfile)

def write_subset(input_file: str, output_file: str, mask: SWIFTMask, restrict: np.ndarray):
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
    restrict : np.ndarray
        length 3 array of spatial ranges (each a 2 element array) containing
        the region of interest
    """
    # Open the files
    infile = h5py.File(input_file, "r")
    outfile = h5py.File(output_file, "w")

    # Write metadata and data subset
    list_of_links, list_of_link_paths = find_links(infile)
    write_metadata(infile, outfile, list_of_links, mask, restrict)
    #update_metadata_counts(outfile, mask, restrict)
    write_datasubset(infile, outfile, mask, find_datasets(infile), list_of_links)
    connect_links(outfile, list_of_links, list_of_link_paths)

    # Clean up
    infile.close()
    outfile.close()
