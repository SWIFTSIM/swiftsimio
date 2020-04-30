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
    if "Cells" in dataset_name:
        return None
    elif "PartType" in dataset_name:
        part_type = [int(x) for x in filter(str.isdigit, dataset_name)][0]
        mask_name = metadata.particle_types.particle_name_underscores[part_type]
        return getattr(mask, f"{mask_name}{suffix}") if hasattr(mask, mask_name) else None
    else:
        return None

def write_datasubset(infile, outfile, mask: SWIFTMask, dataset_names):
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
    """
    if mask is not None:
        for name in dataset_names:
            if "Cells" in name:
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
            subset = read_ranges_from_file(infile[name], dataset_mask, output_shape = output_shape, output_type = output_type)
            
            # Write the subset
            outfile.create_dataset(name, data=subset)

def write_metadata(infile, outfile):
    """
    Copy over all the metadata from snapshot to output file

    Parameters
    ----------
    infile : h5py.File
        hdf5 file handle for input snapshot
    outfile : h5py.File
        hdf5 file handle for output snapshot
    """
    for field in infile.keys():
        if not "PartType" in field:
            infile.copy(field, outfile)

def find_datasets(input_file: h5py.File, dataset_names=[], path = None):
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
    if path is not None:
        keys = input_file[path].keys()
    else:
        keys = input_file.keys()
        path = ""

    for key in keys:
        subpath = f"{path}/{key}"
        if isinstance(input_file[subpath], h5py.Dataset):
            dataset_names.append(subpath)
        elif input_file[subpath].keys() != None:
            find_datasets(input_file, dataset_names, subpath)

    return dataset_names

def write_subset(input_file: str, output_file: str, mask: SWIFTMask):
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
    infile = h5py.File(input_file, "r")
    outfile = h5py.File(output_file, "w")
    
    # Write metadata and data subset
    write_metadata(infile, outfile)
    write_datasubset(infile, outfile, mask, find_datasets(infile))

    # Clean up
    infile.close()
    outfile.close()
