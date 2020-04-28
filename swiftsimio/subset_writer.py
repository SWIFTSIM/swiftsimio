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

class SWIFTDatasubset(object):
    """
    class for reading in subset of SWIFT snapshot
    """

    test_dataset = "PartType0/Coordinates"

    def __init__(self, filename: str, mask: SWIFTMask, mask_size: int):
        # Load the units and metadata
        self.filename = filename
        self.mask = mask
        self.mask_size = mask_size
        self.dataset_names = []
        self.datasets = []
        
        with h5py.File(filename, "r") as infile:
            # Find all the datasets in the snapshot
            infile.visititems(self.find_datasets)

        self.load_datasets()

    def find_datasets(self, name, node):
        #if isinstance(node, h5py.Dataset) and not name in self.dataset_names:
        # ALEXEI: for testing
        if isinstance(node, h5py.Dataset) and not name in self.dataset_names and "Cells" not in name:
            self.dataset_names.append(name)

    def compute_mask_size(self, dataset_name):
        legend = {"Cells": 0, 
                  "PartType0": self.mask.gas_size, 
                  "PartType1": self.mask.dark_matter_size, 
                  "PartType2": 0, 
                  "PartType3": 0, 
                  "PartType4": self.mask.stars_size, 
                  "PartType5": self.mask.black_holes_size}
        for key in legend.keys():
            if key in dataset_name:
                return legend[key]

    def get_mask_label(self, dataset_name):
        legend = {"Cells": 0, 
                  "PartType0": self.mask.gas, 
                  "PartType1": self.mask.dark_matter, 
                  "PartType2": 0, 
                  "PartType3": 0, 
                  "PartType4": self.mask.stars, 
                  "PartType5": self.mask.black_holes}
        for key in legend.keys():
            if key in dataset_name:
                return legend[key]

    def load_datasets(self):
        with h5py.File(self.filename, "r") as infile:
            if self.mask is not None:
                for name in self.dataset_names:
                    # get output dtype and size 
                    first_value = infile[name][0]
                    output_type = first_value.dtype
                    output_size = first_value.size
                    #mask_size = np.sum(self.mask.gas[:,1] - self.mask.gas[:,0])
                    mask_size = self.compute_mask_size(name)
                    if output_size != 1:
                        output_shape = (mask_size, output_size)
                    else:
                        output_shape = mask_size

                    print(name, mask_size, output_shape, output_size, output_type)
                    mask = self.get_mask_label(name)
                    self.datasets.append(read_ranges_from_file(infile[name], mask, output_shape = output_shape, output_type = output_type))



class SWIFTWriterDatasubset(object):
    """
    Class for writing subset of SWIFT snapshot
    """

    def __init__(self, infile: str, outfile: str, mask, mask_size):
        self.input_filename = infile
        self.output_filename = outfile

        self.subset = SWIFTDatasubset(self.input_filename, mask, mask_size)
        self.write_metadata()

    def write_metadata(self):
        """
        Copy over all the metadata from snapshot to output file

        ALEXEI: rewrite this taking advantage of finding the 
        datasets in SWIFTDatasubset
        """
        with h5py.File(self.input_filename, "r") as input_file:
            with h5py.File(self.output_filename, "w") as output_file:
                for field in input_file.keys():
                    if not any([group_str in field for group_str in ["PartType", "Cells"]]):
                        print("copying "+field)
                        input_file.copy(field, output_file)

    def write_datasubset(self):
        """
        Write the SWIFTDatasubset previously read in
        """
        with h5py.File(self.output_filename, "a") as output_file:
            #print(self.subset.dataset_names, len(self.subset.datasets))
            for i in range(len(self.subset.dataset_names)):
                output_file.create_dataset(self.subset.dataset_names[i], data=self.subset.datasets[i])
