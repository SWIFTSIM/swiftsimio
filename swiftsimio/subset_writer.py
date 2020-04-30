"""
Contains functions for reading a subset of a SWIFT dataset and writing
it to a new file.
"""

from swiftsimio.reader import SWIFTUnits, SWIFTMetadata
from swiftsimio.masks import SWIFTMask
from swiftsimio.accelerated import read_ranges_from_file

import unyt
import h5py
import numpy as np

def compute_mask_size(mask: SWIFTMask, dataset_name: str):
    """
    Return number of elements of the specified dataset selected by the mask

    Parameters
    ----------
    mask : SWIFTMask
        the mask used to define subset that is written to new snapshot
    dataset_name : str
        the name of the dataset we're interested in. This is the name from the
        hdf5 file (i.e. "PartType0", rather than "gas")

    Returns
    -------
    int
        size of the subset of the dataset selected by the mask
    """
    legend = {"Cells": 0, 
              "PartType0": mask.gas_size if hasattr(mask, "gas_size") else 0, 
              "PartType1": mask.dark_matter_size if hasattr(mask, "dark_matter_size") else 0, 
              "PartType2": 0, 
              "PartType3": 0, 
              "PartType4": mask.stars_size if hasattr(mask, "stars_size") else 0, 
              "PartType5": mask.black_holes_size if hasattr(mask, "black_holes_size") else 0}
    for key in legend.keys():
        if key in dataset_name:
            return legend[key]

def get_mask_label(mask: SWIFTMask, dataset_name: str):
    """
    Return appropriate mask for appropriate dataset

    Parameters
    ----------
    mask : SWIFTMask
        the mask used to define subset that is written to new snapshot
    dataset_name : str
        the name of the dataset we're interested in. This is the name from the
        hdf5 file (i.e. "PartType0", rather than "gas")

    Returns
    -------
    np.ndarray
        mask for the appropriate dataset

    """
    legend = {"Cells": 0, 
              "PartType0": mask.gas if hasattr(mask, "gas") else 0, 
              "PartType1": mask.dark_matter if hasattr(mask, "dark_matter") else 0, 
              "PartType2": 0, 
              "PartType3": 0, 
              "PartType4": mask.stars if hasattr(mask, "stars") else 0, 
              "PartType5": mask.black_holes if hasattr(mask, "black_holes") else 0}
    for key in legend.keys():
        if key in dataset_name:
            return legend[key]

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
            if "Cells" not in name:
                # get output dtype and size 
                first_value = infile[name][0]
                output_type = first_value.dtype
                output_size = first_value.size
                mask_size = compute_mask_size(mask, name)
                if output_size != 1:
                    output_shape = (mask_size, output_size)
                else:
                    output_shape = mask_size
    
                dataset_mask = get_mask_label(mask, name)
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
        if not any([group_str in field for group_str in ["PartType"]]):
            infile.copy(field, outfile)

def find_datasets(input_file: h5py.File, dataset_names, path = None):
    """
    Recursively finds all the datasets in the snapshot and writes them to a list

    Parameters
    ----------
    input_file : h5py.File
        hdf5 file handle for snapshot
    dataset_names : list of str
        names of datasets found in the snapshot
    path : str, optional
        the path to the current location in the snapshot
    """
    if path != None:
        keys = input_file[path].keys()
    else:
        keys = input_file.keys()
        path = ""

    for key in keys:
        subpath = path + "/" + key
        if isinstance(input_file[subpath], h5py.Dataset):
            dataset_names.append(subpath)
        elif input_file[subpath].keys() != None:
            find_datasets(input_file, dataset_names, subpath)

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

    # Find the datasets
    dataset_names = []
    find_datasets(infile, dataset_names)
    
    # Write metadata and data subset
    write_metadata(infile, outfile)
    write_datasubset(infile, outfile, mask, dataset_names)

    # Clean up
    infile.close()
    outfile.close()
