"""
Contains functions and objects for reading a subset of a SWIFT dataset and writing
it to a new file.
"""

from swiftsimio.reader import SWIFTUnits, SWIFTMetadata
from swiftsimio.masks import SWIFTMask
from swiftsimio.accelerated import read_ranges_from_file

import unyt
import h5py
import numpy as np

# Make everything as standalone functions
def compute_mask_size(mask: SWIFTMask, dataset_name: str):
    legend = {"Cells": 0, 
              "PartType0": mask.gas_size, 
              "PartType1": mask.dark_matter_size, 
              "PartType2": 0, 
              "PartType3": 0, 
              "PartType4": mask.stars_size, 
              "PartType5": mask.black_holes_size}
    for key in legend.keys():
        if key in dataset_name:
            return legend[key]

def get_mask_label(mask: SWIFTMask, dataset_name: str):
    legend = {"Cells": 0, 
              "PartType0": mask.gas, 
              "PartType1": mask.dark_matter, 
              "PartType2": 0, 
              "PartType3": 0, 
              "PartType4": mask.stars, 
              "PartType5": mask.black_holes}
    for key in legend.keys():
        if key in dataset_name:
            return legend[key]

def write_datasubset(infile, outfile, mask: SWIFTMask, dataset_names):
    if mask is not None:
        for name in dataset_names:
            # get output dtype and size 
            first_value = infile[name][0]
            output_type = first_value.dtype
            output_size = first_value.size
            mask_size = compute_mask_size(mask, name)
            if output_size != 1:
                output_shape = (mask_size, output_size)
            else:
                output_shape = mask_size
    
            print(name, mask_size, output_shape, output_size, output_type)
            dataset_mask = get_mask_label(mask, name)
            subset = read_ranges_from_file(infile[name], dataset_mask, output_shape = output_shape, output_type = output_type)
            
            # Write the subset
            print("writing ", name)
            outfile.create_dataset(name, data=subset)

def write_metadata(infile, outfile):
    """
    Copy over all the metadata from snapshot to output file

    ALEXEI: rewrite this taking advantage of finding the 
    datasets in SWIFTDatasubset
    """
    for field in infile.keys():
        if not any([group_str in field for group_str in ["PartType"]]):
            print("copying "+field)
            infile.copy(field, outfile)

def find_datasets(name, node):
    #if isinstance(node, h5py.Dataset) and not name in self.dataset_names:
    # ALEXEI: for testing
    if isinstance(node, h5py.Dataset) and not name in self.dataset_names and "Cells" not in name:
        self.dataset_names.append(name)

def write_subset(input_file: str, output_file: str, mask: SWIFTMask):
    # Open the files
    infile = h5py.File(input_file, "r")
    outfile = h5py.File(output_file, "w")

    # Find the datasets
    #infile.visititems(find_datasets)
    # ALEXEI: temporary for testing, change to using visititems function again
    dataset_names = ["PartType0/Coordinates"]
    
    # Write metadata and data subset
    write_metadata(infile, outfile)
    write_datasubset(infile, outfile, mask, dataset_names)

    # Clean up
    infile.close()
    outfile.close()
